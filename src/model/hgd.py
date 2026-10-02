import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConvLayer
from .gaussian_diffusion import GaussianDiffusion, ModelMeanType
from .DNN import DNN
from .scorefusion import ScoreFusion


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss.mean()


class HGD(nn.Module):

    def __init__(
        self,
        data,
        emb_dim,
        ubg_gnn_layers,
        bsg_gnn_layers,
        diff_gnn_layers,
        tda_layers,
        device=None,
        aux_sample_ratio=10,
        aux_keep_ratio=0.6,  # 新增：每个用户原始辅助行为边保留比例
        aux_keep_min=1,
        diff_loss_weight=0.1,
        diff_sample_steps=100,
        diff_noise_schedule="linear-var",
        diff_noise_scale=0.1,
        diff_noise_min=0.0001,
        diff_noise_max=0.02,
        diff_history_num_per_term=10,
        diff_beta_fixed=True,
        diff_infer_batch_size=256,
        diff_time_emb_dim=10,
        diff_dnn_dropout=0.5,
        diff_dnn_norm=False,
        predict_item_chunk_size=256,
        # 胶囊与后处理超参
        use_momentum_distill=True,
        momentum_tau=0.99,
        md_temperature=0.2,
        lambda_md=0.05,
        lambda_cl=0.1,
        cl_temperature=0.2,
    ):
        super(HGD, self).__init__()
        self.edge_dict = data['edge_dict']
        for key in self.edge_dict:
            if isinstance(self.edge_dict[key], torch.Tensor):
                self.edge_dict[key] = self.edge_dict[key].to(device)
        self.original_edge_dict = {
            behavior_type: self.edge_dict[behavior_type].clone().contiguous()
            for behavior_type in self.edge_dict
        }
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.ubg_gnn_layers = ubg_gnn_layers
        self.bsg_gnn_layers = bsg_gnn_layers
        self.diff_gnn_layers = diff_gnn_layers
        self.tda_layers = tda_layers
        self.device = device
        self.aux_sample_ratio = aux_sample_ratio
        self.aux_keep_ratio = aux_keep_ratio
        self.aux_keep_min = aux_keep_min
        self.diff_loss_weight = diff_loss_weight
        self.diff_sample_steps = diff_sample_steps
        self.diff_noise_schedule = diff_noise_schedule
        self.diff_noise_scale = float(diff_noise_scale)
        self.diff_noise_min = float(diff_noise_min)
        self.diff_noise_max = float(diff_noise_max)
        self.diff_history_num_per_term = int(diff_history_num_per_term)
        self.diff_beta_fixed = bool(diff_beta_fixed)
        self.diff_infer_batch_size = diff_infer_batch_size
        self.diff_time_emb_dim = int(diff_time_emb_dim)
        self.diff_dnn_dropout = float(diff_dnn_dropout)
        self.diff_dnn_norm = bool(diff_dnn_norm)
        self.predict_item_chunk_size = max(1, int(predict_item_chunk_size))
        self.use_momentum_distill = bool(use_momentum_distill)
        self.momentum_tau = float(momentum_tau)
        self.md_temperature = float(md_temperature) if float(md_temperature) > 0 else 0.2
        self.lambda_md = float(lambda_md)
        self.lambda_cl = float(lambda_cl)
        self.cl_temperature = float(cl_temperature) if float(cl_temperature) > 0 else 0.2
        self.bsg_types = data['bsg_types']
        # 目标行为（用于 BPR 与 train.json 中的交互监督）— 由 statistics / main 指定，一般为 buy
        if 'target_behavior' in data:
            self.target_behavior = data['target_behavior']
        elif 'buy' in self.bsg_types:
            self.target_behavior = 'buy'
        else:
            self.target_behavior = self.bsg_types[-1]

        # 胶囊兴趣转移 + 行为聚合的语义锚点：与「购买/转化」语义对齐，默认等于 target_behavior（如 buy）；
        # 若配置里的 target_behavior 不在 bsg_types 中，则退化为 bsg_types[-1]（通常亦为购买行为）。
        if self.target_behavior in self.bsg_types:
            self.semantic_target_behavior = self.target_behavior
        else:
            self.semantic_target_behavior = self.bsg_types[-1]

        # 辅助行为（不含监督目标）：用于扩散与第三层去噪图
        self.aux_types = [b for b in data['aux_types'] if b != self.target_behavior]
        # 去噪图在 data 中无现成边：初始化为对应辅助行为图的拷贝，refresh 后会更新
        for b in self.aux_types:
            dk = f'diff_{b}'
            if dk not in self.edge_dict:
                base = self.edge_dict[b].clone().contiguous()
                self.edge_dict[dk] = base
                self.original_edge_dict[dk] = base.clone().contiguous()

        self.tcb_types = data['tcb_types']
        self.tib_types = data['tib_types']
        self.trbg_types = self.tcb_types + self.tib_types
        self.diff_aux_types = [f'diff_{b}' for b in self.aux_types]
        self.total_behaviors = ['ubg'] + self.bsg_types + self.diff_aux_types


        self.scorefusion = ScoreFusion(
            emb_dim=emb_dim,
            bsg_types=list(self.bsg_types),
            diff_aux_types=list(self.diff_aux_types),
            target_behavior=self.semantic_target_behavior,
        )

        self.bpr_loss = BPRLoss()

        self.user_embedding = nn.Embedding(self.n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)
        # 平滑融合不同 GCN 层，替代 max over layers
        self.convs = nn.ModuleDict()
        self.convs['ubg'] = nn.ModuleList(
            [GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.ubg_gnn_layers)]
        )
        for behavior_type in self.bsg_types:
            self.convs[behavior_type] = nn.ModuleList(
                [GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.bsg_gnn_layers)]
            )
        for behavior_type in self.diff_aux_types:
            self.convs[behavior_type] = nn.ModuleList(
                [GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(self.diff_gnn_layers)]
            )

        # 用户-物品集合：原始行为来自 data；diff_* 必须与当前 edge_dict 一致（去噪图边 = 交集中实际保留的边）
        self.behavior_user_items = dict(data['behavior_user_items'])
        self.behavior_user_degrees = dict(data['behavior_user_degrees'])
        for b in self.aux_types:
            dk = f'diff_{b}'
            self._sync_user_items_from_edges(dk)
        diff_input_dim = self.n_items + 1
        self.aux_diff_models = nn.ModuleDict()
        for behavior_type in self.aux_types:
            self.aux_diff_models[behavior_type] = DNN(
                in_dims=[diff_input_dim, self.emb_dim],
                out_dims=[self.emb_dim, diff_input_dim],
                emb_size=self.diff_time_emb_dim,
                time_type="cat",
                norm=self.diff_dnn_norm,
                dropout=self.diff_dnn_dropout,
            )

        mean_type = ModelMeanType.START_X
        self.diffusion = GaussianDiffusion(
            mean_type=mean_type,
            noise_schedule=self.diff_noise_schedule,
            noise_scale=self.diff_noise_scale,
            noise_min=self.diff_noise_min,
            noise_max=self.diff_noise_max,
            steps=self.diff_sample_steps,
            device=device,
            history_num_per_term=self.diff_history_num_per_term,
            beta_fixed=self.diff_beta_fixed,
        )
        self.reset_parameters()
        if self.use_momentum_distill:
            self._init_momentum_teacher()

    def _edges_to_user_items(self, edge_index):
        """
        从 edge_index 解析 user -> local item id。

        注意：
        当前代码里构造物品节点时使用的是：
            item_node = local_item_id + (n_users + 1)

        所以从全局 item node 反解 local item id 时必须是：
            local_item_id = item_node - (n_users + 1)
        """
        user_items = {}
        n_users, n_items = self.n_users, self.n_items
        item_offset = n_users + 1

        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            u = int(u)
            v = int(v)

            # user id: 1..n_users
            # item node id: item_offset + 1 .. item_offset + n_items
            if 1 <= u <= n_users and item_offset + 1 <= v <= item_offset + n_items:
                item = v - item_offset  # local item id: 1..n_items
                user_items.setdefault(u, set()).add(int(item))

        return user_items

    def _sync_user_items_from_edges(self, behavior_key):
        """使 behavior_user_items[behavior_key] 与当前 self.edge_dict[behavior_key] 一致（用于 diff_*）。"""
        ei = self.edge_dict[behavior_key]
        self.behavior_user_items[behavior_key] = self._edges_to_user_items(ei)
        self.behavior_user_degrees[behavior_key] = {
            u: len(items) for u, items in self.behavior_user_items[behavior_key].items()
        }

    def build_interaction_matrix(self, behavior_type, users=None):
        device = self.user_embedding.weight.device
        behavior_user_items = self.behavior_user_items[behavior_type]
        if users is None:
            mat = torch.zeros(
                self.n_users + 1,
                self.n_items + 1,
                device=device,
                dtype=torch.float32
            )
            users_iter = range(1, self.n_users + 1)
        else:
            users = users.unique()
            mat = torch.zeros(
                users.size(0),
                self.n_items + 1,
                device=device,
                dtype=torch.float32
            )
            users_iter = users.tolist()
        row_users = []
        row_items = []
        for row_idx, user in enumerate(users_iter):
            items = behavior_user_items.get(int(user), set())
            valid_items = [item for item in items if 1 <= item <= self.n_items]
            if valid_items:
                row_users.extend([row_idx if users is not None else user] * len(valid_items))
                row_items.extend(valid_items)
        if row_users:
            row_users = torch.tensor(row_users, device=device, dtype=torch.long)
            row_items = torch.tensor(row_items, device=device, dtype=torch.long)
            mat[row_users, row_items] = 1.0

        return mat

    def merge_edge_index_intersection(self, generated_edge_index, original_edge_index):
        generated_edges = set(map(tuple, generated_edge_index.t().tolist()))
        original_edges = set(map(tuple, original_edge_index.t().tolist()))
        intersect_edges = generated_edges & original_edges
        if len(intersect_edges) == 0:
            return original_edge_index
        intersect_edges = torch.tensor(
            list(intersect_edges),
            device=generated_edge_index.device,
            dtype=torch.long
        ).t().contiguous()
        return intersect_edges

    def select_existing_edges_by_diff_scores(
            self,
            diff_interaction,
            behavior_type,
            keep_ratio=None,
            keep_min=None,
            users=None,
    ):
        """
        只在用户原始辅助行为已有物品集合中打分，然后保留 top-p。

        diff_interaction:
            users is None 时: [n_users + 1, n_items + 1]
            users 不为空时:  [len(users), n_items + 1]

        返回:
            sampled_user_item_edges: [2, E]
            第 0 行是 user id，第 1 行是 local item id。
        """
        device = diff_interaction.device
        keep_ratio = self.aux_keep_ratio if keep_ratio is None else keep_ratio
        keep_min = self.aux_keep_min if keep_min is None else keep_min
        if users is None:
            users_iter = list(range(1, self.n_users + 1))
        else:
            users = users.unique()
            users_iter = users.tolist()
        user_items = self.behavior_user_items[behavior_type]
        kept_edges = []
        for row_idx, user in enumerate(users_iter):
            cand_items = sorted(list(user_items.get(int(user), set())))
            cand_items = [i for i in cand_items if 1 <= i <= self.n_items]

            if len(cand_items) == 0:
                continue

            cand = torch.tensor(cand_items, device=device, dtype=torch.long)

            # diff_interaction 的列索引就是 local item id，0 是 padding item
            if users is None:
                score_row = diff_interaction[int(user)]
            else:
                score_row = diff_interaction[row_idx]

            scores = score_row[cand]

            k = int(math.ceil(keep_ratio * len(cand_items)))
            k = max(int(keep_min), k)
            k = min(k, len(cand_items))

            top_idx = torch.topk(scores, k=k, dim=0).indices
            selected_items = cand[top_idx]

            user_nodes = torch.full(
                (selected_items.numel(),),
                int(user),
                device=device,
                dtype=torch.long,
            )

            kept_edges.append(torch.stack([user_nodes, selected_items], dim=0))
        if len(kept_edges) == 0:
            return None
        return torch.cat(kept_edges, dim=1)

    def filter_sampled_edges_by_original_graph(self, sampled_user_item_edges, behavior_type):
        if sampled_user_item_edges is None:
            return None
        device = sampled_user_item_edges.device
        user_items = self.behavior_user_items[behavior_type]
        kept_edges = []
        for user, item in sampled_user_item_edges.t().tolist():
            if item in user_items.get(int(user), set()):
                kept_edges.append((user, item))
        if not kept_edges:
            return None
        return torch.tensor(kept_edges, device=device, dtype=torch.long).t().contiguous()

    def set_denoised_aux_graph_from_filtered_edges(self, behavior_type, sampled_user_item_edges):
        device = self.user_embedding.weight.device
        diff_behavior_type = f'diff_{behavior_type}'
        old_edges = self.edge_dict.get(diff_behavior_type)
        if sampled_user_item_edges is None:
            self.edge_dict[diff_behavior_type] = self.original_edge_dict[behavior_type].to(device)
        else:
            user_nodes = sampled_user_item_edges[0]
            local_item_ids = sampled_user_item_edges[1]
            item_nodes = local_item_ids + (self.n_users + 1)
            self.edge_dict[diff_behavior_type] = torch.stack([
                torch.cat([user_nodes, item_nodes]),
                torch.cat([item_nodes, user_nodes])
            ], dim=0)
        del old_edges
        # 胶囊邻居必须与当前去噪图边一致（过滤后留下的 user-item 即生成∩原始的交集边）
        self._sync_user_items_from_edges(diff_behavior_type)

    def update_denoised_aux_graph(self, behavior_type, diff_interaction):
        """
        根据扩散模型输出，只在原始辅助行为已有边中选择 top-p 边，构造 diff_* 去噪图。
        不再做“全物品采样 -> 与原图求交”。
        """
        device = diff_interaction.device
        diff_behavior_type = f'diff_{behavior_type}'

        sampled_user_item_edges = self.select_existing_edges_by_diff_scores(
            diff_interaction=diff_interaction,
            behavior_type=behavior_type,
            keep_ratio=self.aux_keep_ratio,
            keep_min=self.aux_keep_min,
            users=None,
        )

        self.set_denoised_aux_graph_from_filtered_edges(
            behavior_type=behavior_type,
            sampled_user_item_edges=sampled_user_item_edges,
        )

    @torch.no_grad()
    def refresh_unified_graph(self):
        for behavior_type in self.aux_types:
            diff_model = self.aux_diff_models[behavior_type]
            was_training = diff_model.training
            diff_model.eval()
            sampled_chunks = []
            chunk_size = max(1, int(self.diff_infer_batch_size))
            for start_user in range(1, self.n_users + 1, chunk_size):
                end_user = min(start_user + chunk_size - 1, self.n_users)
                users = torch.arange(
                    start_user,
                    end_user + 1,
                    device=self.user_embedding.weight.device,
                    dtype=torch.long
                )
                x_start = self.build_interaction_matrix(behavior_type, users)
                diff_interaction = self.diffusion.p_sample(
                    model=diff_model,
                    x_start=x_start,
                    steps=self.diff_sample_steps,
                    sampling_noise=False
                )
                sampled_user_item_edges = self.select_existing_edges_by_diff_scores(
                    diff_interaction=diff_interaction,
                    behavior_type=behavior_type,
                    keep_ratio=self.aux_keep_ratio,
                    keep_min=self.aux_keep_min,
                    users=users,
                )
                if sampled_user_item_edges is not None:
                    sampled_chunks.append(sampled_user_item_edges)
                del x_start, diff_interaction
            if was_training:
                diff_model.train()
            if sampled_chunks:
                sampled_user_item_edges = torch.cat(sampled_chunks, dim=1)
            else:
                sampled_user_item_edges = None
            del sampled_chunks
            self.set_denoised_aux_graph_from_filtered_edges(behavior_type, sampled_user_item_edges)
            del sampled_user_item_edges

    def compute_diffusion_loss(self, users):
        total_diff_loss = 0.0
        for behavior_type in self.aux_types:
            x_start = self.build_interaction_matrix(behavior_type, users)
            diff_terms = self.diffusion.training_losses(
                model=self.aux_diff_models[behavior_type],
                x_start=x_start,
                reweight=False
            )
            total_diff_loss = total_diff_loss + diff_terms["loss"].mean()
        return total_diff_loss

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.scorefusion.reset_parameters()

    def _init_momentum_teacher(self):
        # Momentum distillation teacher for auxiliary-to-target behavior embeddings.
        # This EMA teacher only mirrors the graph embedding modules.
        self.teacher_user_embedding = copy.deepcopy(self.user_embedding)
        self.teacher_item_embedding = copy.deepcopy(self.item_embedding)
        self.teacher_convs = copy.deepcopy(self.convs)
        for module in [self.teacher_user_embedding, self.teacher_item_embedding, self.teacher_convs]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def update_momentum_teacher(self):
        if not self.use_momentum_distill:
            return
        tau = self.momentum_tau
        with torch.no_grad():
            for teacher_param, student_param in zip(
                    self.teacher_user_embedding.parameters(),
                    self.user_embedding.parameters()):
                teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)
            for teacher_param, student_param in zip(
                    self.teacher_item_embedding.parameters(),
                    self.item_embedding.parameters()):
                teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)
            for teacher_param, student_param in zip(self.teacher_convs.parameters(), self.convs.parameters()):
                teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)

    def propagate(self, x, edge_index, behavior_type, target_emb=None, convs=None):
        """单层循环求和形式的前向（兼容保留）；x: [N, D] -> [N, D]"""
        convs = self.convs if convs is None else convs
        result = [x]
        cur = x
        for i, conv in enumerate(convs[behavior_type]):
            cur = conv(cur, edge_index, target_emb)
            cur = F.normalize(cur, dim=1)
            result.append(cur / (i + 1))
        result = torch.stack(result, dim=1)
        cur = result.sum(dim=1)
        return cur

    def propagate_all_layers(self, x, edge_index, behavior_type, target_emb=None, convs=None):
        """
        返回每一图卷积层（含初始嵌入）的节点表示，不做跨层求和。
        x: [N, D]
        返回: [N, L, D]，其中 L = gnn_layers + 1
        """
        convs = self.convs if convs is None else convs
        layers = [x]
        cur = x
        for i, conv in enumerate(convs[behavior_type]):
            cur = conv(cur, edge_index, target_emb)
            cur = F.normalize(cur, dim=1)
            layers.append(cur / (i + 1))
        return torch.stack(layers, dim=1)

    def _encode_graph_embeddings(self, user_weight, item_weight, convs):
        edge_dict = self.edge_dict
        init_emb = torch.cat([user_weight, item_weight], dim=0)

        ubg_flat = self.propagate(init_emb, edge_dict['ubg'], 'ubg', convs=convs)

        layer_embs = {}
        flat_embs = {'ubg': ubg_flat}

        for behavior_type in self.bsg_types:
            stack = self.propagate_all_layers(ubg_flat, edge_dict[behavior_type], behavior_type, convs=convs)
            layer_embs[behavior_type] = stack
            flat_embs[behavior_type] = stack.sum(dim=1)

        for behavior_type in self.aux_types:
            diff_key = f'diff_{behavior_type}'
            stack = self.propagate_all_layers(
                ubg_flat,
                edge_dict[diff_key],
                diff_key,
                convs=convs,
            )
            layer_embs[diff_key] = stack
            flat_embs[diff_key] = stack.sum(dim=1)

        return flat_embs, layer_embs

    def forward(self, train=False):
        flat_embs, _ = self._encode_graph_embeddings(
            self.user_embedding.weight,
            self.item_embedding.weight,
            self.convs,
        )
        # flat_embs already equals layer stack sum; omit layer_embs to reduce autograd retention.
        return dict(flat_embs)

    @torch.no_grad()
    def teacher_forward(self):
        if not self.use_momentum_distill:
            return None
        self.teacher_user_embedding.eval()
        self.teacher_item_embedding.eval()
        self.teacher_convs.eval()
        flat_embs, _ = self._encode_graph_embeddings(
            self.teacher_user_embedding.weight,
            self.teacher_item_embedding.weight,
            self.teacher_convs,
        )
        return flat_embs

    def bidirectional_kl_distill(self, student_a, student_b, teacher_a, teacher_b, temperature):
        temperature = temperature if temperature > 0 else 0.2
        student_a = F.normalize(student_a, p=2, dim=-1, eps=1e-12)
        student_b = F.normalize(student_b, p=2, dim=-1, eps=1e-12)
        teacher_a = F.normalize(teacher_a, p=2, dim=-1, eps=1e-12)
        teacher_b = F.normalize(teacher_b, p=2, dim=-1, eps=1e-12)

        student_ab = student_a @ student_b.t() / temperature
        student_ba = student_b @ student_a.t() / temperature
        with torch.no_grad():
            teacher_ab = teacher_a @ teacher_b.t() / temperature
            teacher_ba = teacher_b @ teacher_a.t() / temperature
            target_ab = F.softmax(teacher_ab, dim=1)
            target_ba = F.softmax(teacher_ba, dim=1)

        loss_ab = F.kl_div(F.log_softmax(student_ab, dim=1), target_ab, reduction='batchmean')
        loss_ba = F.kl_div(F.log_softmax(student_ba, dim=1), target_ba, reduction='batchmean')
        return loss_ab + loss_ba

    def compute_momentum_distillation_loss(self, student_emb_dict, users, items, teacher_emb_dict=None):
        if (not self.use_momentum_distill) or (not self.training):
            return self.user_embedding.weight.new_zeros(())

        if teacher_emb_dict is None:
            teacher_emb_dict = self.teacher_forward()
        target_key = self.semantic_target_behavior
        if teacher_emb_dict is None or target_key not in student_emb_dict or target_key not in teacher_emb_dict:
            return self.user_embedding.weight.new_zeros(())

        users = users.unique()
        items = items.unique()
        items = items[(items >= 1) & (items <= self.n_items)]
        if users.numel() == 0 or items.numel() == 0:
            return self.user_embedding.weight.new_zeros(())

        item_nodes = items + self.n_users + 1
        student_target_user = student_emb_dict[target_key][users]
        student_target_item = student_emb_dict[target_key][item_nodes]
        teacher_target_user = teacher_emb_dict[target_key][users]
        teacher_target_item = teacher_emb_dict[target_key][item_nodes]

        aux_behavior_keys = [b for b in self.bsg_types if b != target_key] + list(self.diff_aux_types)
        md_loss = self.user_embedding.weight.new_zeros(())
        for behavior_type in aux_behavior_keys:
            if behavior_type not in student_emb_dict or behavior_type not in teacher_emb_dict:
                continue
            student_aux_user = student_emb_dict[behavior_type][users]
            student_aux_item = student_emb_dict[behavior_type][item_nodes]
            teacher_aux_user = teacher_emb_dict[behavior_type][users]
            teacher_aux_item = teacher_emb_dict[behavior_type][item_nodes]

            md_loss = md_loss + self.bidirectional_kl_distill(
                student_aux_user,
                student_target_user,
                teacher_aux_user,
                teacher_target_user,
                self.md_temperature,
            )
            md_loss = md_loss + self.bidirectional_kl_distill(
                student_aux_item,
                student_target_item,
                teacher_aux_item,
                teacher_target_item,
                self.md_temperature,
            )

        return md_loss

    @staticmethod
    def infonce_contrastive_loss(anchor, positive, temperature):
        """
        InfoNCE on a batch: (anchor[i], positive[i]) is positive;
        (anchor[i], positive[j]), i!=j is negative.
        anchor/positive: [B, D]
        """
        temperature = temperature if temperature > 0 else 0.2
        anchor = F.normalize(anchor, p=2, dim=-1, eps=1e-12)
        positive = F.normalize(positive, p=2, dim=-1, eps=1e-12)
        pos_score = torch.exp((anchor * positive).sum(dim=-1) / temperature)
        logits = anchor @ positive.t() / temperature
        ttl_score = torch.exp(logits).sum(dim=1)
        return (-torch.log(pos_score / (ttl_score + 1e-12))).mean()

    def _filter_batch_nodes(self, users, items):
        users = users.unique()
        items = items.unique()
        items = items[(items >= 1) & (items <= self.n_items)]
        if users.numel() == 0 or items.numel() == 0:
            return None, None, None
        item_nodes = items + self.n_users + 1
        return users, items, item_nodes

    def compute_aux_target_contrastive_loss(self, emb_dict, users, items, aux_behavior_keys):
        """
        Contrastive learning: same user's auxiliary vs target embedding is positive,
        different users/items in batch are negatives. Applied on user and item sides.
        """
        if self.lambda_cl <= 0:
            return self.user_embedding.weight.new_zeros(())

        target_key = self.semantic_target_behavior
        if target_key not in emb_dict:
            return self.user_embedding.weight.new_zeros(())

        users, _, item_nodes = self._filter_batch_nodes(users, items)
        if users is None:
            return self.user_embedding.weight.new_zeros(())

        target_user = emb_dict[target_key][users]
        target_item = emb_dict[target_key][item_nodes]
        cl_loss = self.user_embedding.weight.new_zeros(())
        n_terms = 0
        for aux_key in aux_behavior_keys:
            if aux_key not in emb_dict:
                continue
            aux_user = emb_dict[aux_key][users]
            aux_item = emb_dict[aux_key][item_nodes]
            cl_loss = cl_loss + self.infonce_contrastive_loss(
                aux_user, target_user, self.cl_temperature
            )
            cl_loss = cl_loss + self.infonce_contrastive_loss(
                aux_item, target_item, self.cl_temperature
            )
            n_terms += 2

        if n_terms == 0:
            return self.user_embedding.weight.new_zeros(())
        return cl_loss / n_terms

    def compute_contrastive_loss(self, emb_dict, users, items):
        cl_bsg = self.compute_aux_target_contrastive_loss(
            emb_dict, users, items, self.aux_types
        )
        cl_diff = self.compute_aux_target_contrastive_loss(
            emb_dict, users, items, self.diff_aux_types
        )
        return cl_bsg + cl_diff

    def score_pairs(self, emb_dict, users, items):
        """
        score fusion: target score + beta1 * gated aux scores + beta2 * gated diff scores.
        users/items: [B]
        """
        return self.scorefusion.score_pairs(emb_dict, users, items, self.n_users)

    def loss(self, users, pos_idx, neg_idx):
        emb_dict = self.forward(train=True)
        p_score = self.score_pairs(emb_dict, users, pos_idx)
        n_score = self.score_pairs(emb_dict, users, neg_idx)
        rec_loss = self.bpr_loss(p_score, n_score)
        md_items = torch.cat([pos_idx, neg_idx], dim=0)
        zero = rec_loss.new_zeros(())

        if self.diff_loss_weight > 0:
            diff_loss = self.compute_diffusion_loss(users)
        else:
            diff_loss = zero

        if self.use_momentum_distill and self.lambda_md > 0 and self.training:
            teacher_emb_dict = self.teacher_forward()
            md_loss = self.compute_momentum_distillation_loss(
                emb_dict, users, md_items, teacher_emb_dict=teacher_emb_dict
            )
        else:
            md_loss = zero

        if self.lambda_cl > 0:
            cl_loss = self.compute_contrastive_loss(emb_dict, users, md_items)
        else:
            cl_loss = zero

        return (
            rec_loss
            + self.diff_loss_weight * diff_loss
            + self.lambda_md * md_loss
            + self.lambda_cl * cl_loss
        )

    def predict(self, users, mask_train_items: bool = False):
        emb_dict = self.forward(train=False)
        scores = self._predict_pairwise_scores(
            emb_dict, users, item_chunk_size=self.predict_item_chunk_size
        )
        # 1. 永远屏蔽 padding item
        scores[:, 0] = -1e9
        # 2. 可选：屏蔽训练集中已交互过的目标行为物品
        if mask_train_items:
            target_user_items = self.behavior_user_items.get(self.target_behavior, {})
            for row, uid in enumerate(users.tolist()):
                seen_items = target_user_items.get(int(uid), set())
                seen_items = [i for i in seen_items if 1 <= i <= self.n_items]
                if len(seen_items) > 0:
                    idx = torch.tensor(seen_items, device=scores.device, dtype=torch.long)
                    scores[row, idx] = -1e9
        return scores

    def _predict_pairwise_scores(self, emb_dict, users, item_chunk_size=1024):
        device = users.device
        all_items = torch.arange(0, self.n_items + 1, device=device, dtype=torch.long)
        scores = torch.empty(users.size(0), self.n_items + 1, device=device)

        for start in range(0, self.n_items + 1, item_chunk_size):
            items = all_items[start:start + item_chunk_size]
            pair_users = users.repeat_interleave(items.numel())
            pair_items = items.repeat(users.size(0))
            pair_scores = self.score_pairs(emb_dict, pair_users, pair_items)
            scores[:, start:start + items.numel()] = pair_scores.view(users.size(0), items.numel())

        return scores
