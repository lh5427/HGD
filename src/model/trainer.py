import gc
import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from .metrics import ndcg, hit, recall


def _maybe_empty_cuda_cache(device):
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.eval_topks = sorted(set(args.eval_topks + [10, 20]))
        self.topk = max(args.topk, max(self.eval_topks))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']
        with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices = user_indices.to(self.args.device, non_blocking=True)
                pos_indices = pos_indices.to(self.args.device, non_blocking=True)
                neg_indices = neg_indices.to(self.args.device, non_blocking=True)
                loss = self.model.loss(user_indices, pos_indices, neg_indices)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if hasattr(self.model, 'update_momentum_teacher'):
                    self.model.update_momentum_teacher()
                pbar.set_description(f'Batch Loss: {loss.item():.4f}')
                pbar.update()
                total_loss += loss.item()
                del loss
        return total_loss / len(pbar)

    def train_model(self):
        num_epochs = self.args.num_epochs
        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', leave=False)
        for epoch in pbar:
            self.model.refresh_unified_graph()
            _maybe_empty_cuda_cache(self.args.device)
            loss = self.train_epoch(epoch)
            pbar.set_description(f'Epoch {epoch+1} total loss: {loss:.4f}')
            pbar.update()
            _maybe_empty_cuda_cache(self.args.device)
            gc.collect()

        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), f'{checkpoint_dir}/model.pt')

    def evaluate(self):
        device = self.args.device
        self.model.eval()
        topk_list = []
        with torch.no_grad():
            for user_indices in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices = user_indices.to(device, non_blocking=True)
                scores = self.model.predict(user_indices)
                scores_cpu = scores.cpu()
                del scores
                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores_cpu[user_idx, train_items] = -np.inf

                _, topk_indices = torch.topk(scores_cpu, self.topk, dim=1)
                for idx, user in enumerate(user_indices):
                    gt_items = np.array(self.data['test_gt'][str(user.item())])
                    topk_items = topk_indices[idx].numpy()
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)
                del scores_cpu, topk_indices

        _maybe_empty_cuda_cache(device)
        topk_list = np.vstack(topk_list)
        if topk_list.shape[0] != self.data['test_gt_length'].shape[0]:
            raise ValueError(
                f"Evaluation user count mismatch: got {topk_list.shape[0]} score rows, "
                f"but {self.data['test_gt_length'].shape[0]} ground-truth rows."
            )
        hr_res = hit(topk_list, self.data['test_gt_length']).mean(axis=0)
        recall_res = recall(topk_list, self.data['test_gt_length']).mean(axis=0)
        ndcg_res = ndcg(topk_list, self.data['test_gt_length']).mean(axis=0)

        metrics = {}
        for k in self.eval_topks:
            metrics[f'HR@{k}'] = hr_res[k - 1]
            metrics[f'Recall@{k}'] = recall_res[k - 1]
            metrics[f'NDCG@{k}'] = ndcg_res[k - 1]

        return metrics
