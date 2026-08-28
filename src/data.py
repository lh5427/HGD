import os
import torch
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader 
from loguru import logger


class BPRDataset(Dataset):
    def __init__(self, buy_interactions, n_items):
        self.buy_interactions = {int(k):set(v) for k,v in buy_interactions.items()}
        self.all_items = set(range(1, n_items+1))
        self.users = list(self.buy_interactions.keys())
        self.total_samples = []
        
        for user in self.users:
            for pos_item in self.buy_interactions[user]:
                self.total_samples.append((user, pos_item))
                
    def __len__(self):
        return len(self.total_samples)

    def __getitem__(self, idx):
        user, pos_item = self.total_samples[idx]
        neg_candidates = list(self.all_items - self.buy_interactions[user])
        neg_item = random.choice(neg_candidates)
        
        return torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(neg_item, dtype=torch.long)


class UserDataset(Dataset):
    def __init__(self, users):
        self.users = [int(user) for user in users]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return torch.tensor(self.users[idx], dtype=torch.long)


def convert_edge(edge_list, n_users):
    # Item index starts from n_users+1
    edge_list[1] += n_users + 1
    # Add reverse item to user edges 
    edge_list = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    return edge_list
    
# def build_interaction_matrix(edge_index, n_users, n_items, device):
#     """
#     Build user-item interaction matrix from ubg edges
#     """
#     user_item_mat = torch.zeros(n_users + 1, n_items + 1, device=device)
#     src, dst = edge_index
#     for u, v in zip(src, dst):
#         if u <= n_users and v > n_users:
#             item = v - n_users
#             user_item_mat[u, item] = 1.0
#     return user_item_mat

def load_data(data_dir, dataset, device, batch_size, eval_batch_size=None):
    logger.info('Load data')
    data_dir = os.path.join(data_dir, dataset)
    
    # load data statistics
    with open(f'{data_dir}/statistics.json', 'r') as f:
        statistics = json.load(f)
        
    n_users, n_items = statistics['n_users'], statistics['n_items']
    bsg_types, tcb_types, tib_types = statistics['bsg_types'], statistics['tcb_types'], statistics['tib_types']

    target_behavior = statistics.get('target_behavior')
    if target_behavior is None:
        target_behavior = 'buy' if 'buy' in bsg_types else bsg_types[-1]
    statistics['target_behavior'] = target_behavior
    
    # load edges
    edge_dict = dict()
    # for behavior_type in ['ubg'] + bsg_types + trbg_types:
    for behavior_type in ['ubg'] + bsg_types:
        edge_list = np.loadtxt(f'{data_dir}/{behavior_type}.txt', dtype=int)
        edge_list = torch.from_numpy(edge_list).to(device).T
        edge_dict[behavior_type] = convert_edge(edge_list, n_users)
    
    # load train/test buy interactions
    with open(f'{data_dir}/train.json', 'r') as f:
        train_buy = json.load(f)
    with open(f'{data_dir}/test.json', 'r') as f:
        test_buy = json.load(f)    
    train_dataset = BPRDataset(train_buy, n_items)
    test_users = [int(user) for user in test_buy.keys()]
    test_dataset = UserDataset(test_users)
    if eval_batch_size is None:
        eval_batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )
    test_gt_length = np.array([len(test_buy[str(user)]) for user in test_users])
    
    data = dict()
    data['edge_dict'] = edge_dict
    data['train_loader'] = train_loader
    data['test_loader'] = test_loader
    data['train_gt'] = train_buy
    data['test_gt'] = test_buy
    data['test_users'] = test_users
    data['test_gt_length'] = test_gt_length
    data.update(statistics)

    # ===== build user interaction dicts for diffusion =====
    behavior_user_items = dict()
    behavior_user_degrees = dict()
    behavior_keys_for_neighbors = list(dict.fromkeys(['ubg'] + bsg_types + statistics['aux_types']))
    item_offset = n_users + 1
    for behavior_type in behavior_keys_for_neighbors:
        user_items = dict()
        edge_index = edge_dict[behavior_type]
        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            if 1 <= u <= n_users and item_offset + 1 <= v <= item_offset + n_items:
                item = v - item_offset
                user_items.setdefault(u, set()).add(item)
        behavior_user_items[behavior_type] = user_items
        behavior_user_degrees[behavior_type] = {
            user: len(items) for user, items in user_items.items()
        }

    data['behavior_user_items'] = behavior_user_items
    data['behavior_user_degrees'] = behavior_user_degrees
    # 兼容旧字段
    data['ubg_user_items'] = behavior_user_items['ubg']
    data['ubg_user_degrees'] = behavior_user_degrees['ubg']


    return data
    
