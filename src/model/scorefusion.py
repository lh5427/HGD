import torch
import torch.nn as nn


class ScoreFusionGating(nn.Module):
    """
    GAT-style attention over auxiliary behavior scores.
    Each auxiliary score is weighted by softmax(LeakyReLU(W @ [target_expert || aux_expert] + b)).
    target_expert = target_user * target_item (element-wise)
    aux_expert = aux_user * aux_item (element-wise)
    """

    def __init__(self, emb_dim, aux_behavior_types):
        super(ScoreFusionGating, self).__init__()
        self.aux_behavior_types = list(aux_behavior_types)
        self.att = nn.Linear(emb_dim * 2, 1, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att.weight)
        if self.att.bias is not None:
            nn.init.zeros_(self.att.bias)

    def _pair_expert(self, user_emb, item_emb):
        """user_emb/item_emb: [B, D] -> expert [B, D]"""
        return user_emb * item_emb

    def _compute_weights(self, target_expert, aux_experts):
        """
        target_expert: [..., D]
        aux_experts: list of [..., D], len = num_aux
        returns: [..., num_aux]
        """
        logits = []
        for aux_expert in aux_experts:
            gate_input = torch.cat([target_expert, aux_expert], dim=-1)
            logit = self.leaky_relu(self.att(gate_input)).squeeze(-1)
            logits.append(logit)
        return torch.softmax(torch.stack(logits, dim=-1), dim=-1)

    def fuse_pair_scores(self, target_score, aux_scores, target_expert, aux_experts):
        """
        target_score: [B]
        aux_scores: list of [B]
        target_expert / aux_experts: [B, D]
        returns fused correction: [B]  (beta applied outside)
        """
        if len(aux_scores) == 0:
            return target_score.new_zeros(target_score.shape)
        weights = self._compute_weights(target_expert, aux_experts)
        aux_stack = torch.stack(aux_scores, dim=-1)
        return (weights * aux_stack).sum(dim=-1)

    def fuse_all_item_scores(self, target_score, aux_score_list, target_expert, aux_expert_list):
        """
        Full-matrix variant for predict.
        target_score: [B, N]
        aux_score_list: list of [B, N]
        target_expert / aux_expert_list: [B, N, D]
        returns correction: [B, N]
        """
        if len(aux_score_list) == 0:
            return target_score.new_zeros(target_score.shape)
        weights = self._compute_weights(target_expert, aux_expert_list)
        aux_stack = torch.stack(aux_score_list, dim=-1)
        return (weights * aux_stack).sum(dim=-1)


class ScoreFusion(nn.Module):
    """
    Multi-behavior score fusion (no embedding fusion).

    final = s_target
            + beta1 * sum_{aux in layer-2} (alpha_aux * s_aux)
            + beta2 * sum_{diff in layer-3} (gamma_diff * s_diff)

    Target score weight is fixed at 1. alpha/gamma are GAT-style softmax weights
    computed from element-wise user-item experts per pair.
    """

    def __init__(self, emb_dim, bsg_types, diff_aux_types, target_behavior='buy'):
        super(ScoreFusion, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = list(bsg_types)
        self.diff_aux_types = list(diff_aux_types)
        self.diff_aux_by_behavior = {
            behavior_type[len('diff_'):]: behavior_type
            for behavior_type in self.diff_aux_types
            if behavior_type.startswith('diff_')
        }
        self.target_behavior = target_behavior
        if target_behavior not in self.bsg_types:
            raise ValueError(f"target_behavior {target_behavior} must be in bsg_types")

        self.aux_bsg_types = [b for b in self.bsg_types if b != target_behavior]
        self.aux_diff_types = [
            self.diff_aux_by_behavior[b]
            for b in self.aux_bsg_types
            if b in self.diff_aux_by_behavior
        ]

        self.bsg_gate = ScoreFusionGating(emb_dim, self.aux_bsg_types)
        self.diff_gate = ScoreFusionGating(emb_dim, self.aux_diff_types)
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta2 = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.bsg_gate.reset_parameters()
        self.diff_gate.reset_parameters()
        nn.init.constant_(self.beta1, 1.0)
        nn.init.constant_(self.beta2, 1.0)

    def _select_behavior_embedding(self, emb_dict, behavior_type):
        layer_embs = emb_dict.get('layer_embs')
        if layer_embs is not None and behavior_type in layer_embs:
            return layer_embs[behavior_type].sum(dim=1)
        return emb_dict[behavior_type]

    def _split_user_item(self, node_emb, n_users):
        user_emb = node_emb[:n_users + 1]
        item_emb = node_emb[n_users + 1:]
        return user_emb, item_emb

    def _collect_behavior_embeddings(self, emb_dict, n_users):
        user_embs = {}
        item_embs = {}
        for behavior_type in self.bsg_types:
            node_emb = self._select_behavior_embedding(emb_dict, behavior_type)
            u, i = self._split_user_item(node_emb, n_users)
            user_embs[behavior_type] = u
            item_embs[behavior_type] = i

        for behavior_type in self.aux_bsg_types:
            diff_key = self.diff_aux_by_behavior.get(behavior_type, f'diff_{behavior_type}')
            if diff_key not in emb_dict:
                raise KeyError(f"Missing denoised auxiliary behavior embedding: {diff_key}")
            node_emb = self._select_behavior_embedding(emb_dict, diff_key)
            u, i = self._split_user_item(node_emb, n_users)
            user_embs[diff_key] = u
            item_embs[diff_key] = i

        return user_embs, item_embs

    @staticmethod
    def _behavior_dot_score(user_emb, item_emb, users, items):
        """Inner product score for (users, items) pairs. users/items: [B]"""
        u = user_emb[users]
        i = item_emb[items]
        return (u * i).sum(dim=-1)

    def _fuse_pair_scores_from_embs(self, user_embs, item_embs, users, items):
        target_key = self.target_behavior
        target_u = user_embs[target_key][users]
        target_i = item_embs[target_key][items]
        target_score = (target_u * target_i).sum(dim=-1)
        target_expert = target_u * target_i

        aux_scores, aux_experts = [], []
        for behavior_type in self.aux_bsg_types:
            u = user_embs[behavior_type][users]
            i = item_embs[behavior_type][items]
            aux_scores.append((u * i).sum(dim=-1))
            aux_experts.append(u * i)

        diff_scores, diff_experts = [], []
        for diff_key in self.aux_diff_types:
            u = user_embs[diff_key][users]
            i = item_embs[diff_key][items]
            diff_scores.append((u * i).sum(dim=-1))
            diff_experts.append(u * i)

        bsg_correction = self.bsg_gate.fuse_pair_scores(
            target_score, aux_scores, target_expert, aux_experts
        )
        diff_correction = self.diff_gate.fuse_pair_scores(
            target_score, diff_scores, target_expert, diff_experts
        )

        final_score = target_score + self.beta1 * bsg_correction + self.beta2 * diff_correction
        return final_score

    def score_pairs(self, emb_dict, users, items, n_users):
        user_embs, item_embs = self._collect_behavior_embeddings(emb_dict, n_users)
        return self._fuse_pair_scores_from_embs(user_embs, item_embs, users, items)

    def predict_scores(self, emb_dict, users, n_users):
        """
        Fused scores for users over all items: [len(users), n_items + 1]
        """
        user_embs, item_embs = self._collect_behavior_embeddings(emb_dict, n_users)
        device = users.device
        n_items = item_embs[self.target_behavior].size(0) - 1
        all_items = torch.arange(0, n_items + 1, device=device, dtype=torch.long)

        target_key = self.target_behavior
        target_u = user_embs[target_key][users]
        target_i = item_embs[target_key]

        target_scores = target_u @ target_i.t()
        target_expert = target_u.unsqueeze(1) * target_i.unsqueeze(0)

        aux_score_list, aux_expert_list = [], []
        for behavior_type in self.aux_bsg_types:
            u = user_embs[behavior_type][users]
            i = item_embs[behavior_type]
            aux_score_list.append(u @ i.t())
            aux_expert_list.append(u.unsqueeze(1) * i.unsqueeze(0))

        diff_score_list, diff_expert_list = [], []
        for diff_key in self.aux_diff_types:
            u = user_embs[diff_key][users]
            i = item_embs[diff_key]
            diff_score_list.append(u @ i.t())
            diff_expert_list.append(u.unsqueeze(1) * i.unsqueeze(0))

        bsg_correction = self.bsg_gate.fuse_all_item_scores(
            target_scores, aux_score_list, target_expert, aux_expert_list
        )
        diff_correction = self.diff_gate.fuse_all_item_scores(
            target_scores, diff_score_list, target_expert, diff_expert_list
        )

        return target_scores + self.beta1 * bsg_correction + self.beta2 * diff_correction
