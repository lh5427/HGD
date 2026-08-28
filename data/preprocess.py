import json


DATASET_BEHAVIORS = {
    'taobao': ['view', 'cart'],
    'tmall': ['view', 'cart', 'collect'],
    'jdata': ['view', 'cart', 'collect'],
    'tmall1': ['view', 'collect', 'cart'],
    'retail_rocket': ['view', 'cart'],
    'ijcai_15': ['view', 'collect', 'cart'],
}


def list2set(l):
    return set(l)

def list2dict(l, n_users):
    user_interaction_dict = dict()
    for user, item in l:
        if user not in user_interaction_dict.keys():
            user_interaction_dict[user] = []
        user_interaction_dict[user].append(item)
    return user_interaction_dict

def load_edges(path):
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user, item = line.split()[:2]
            edges.append((int(user), int(item)))
    return edges

def save_edges(path, edges):
    with open(path, 'w') as f:
        for user, item in edges:
            f.write(f'{user} {item}\n')

def preprocess():
    for data, aux_behaviors in DATASET_BEHAVIORS.items():
        print(f'Preprocessing {data}...')
        
        # Load behavior-specific graph
        behavior_edges = {
            behavior: load_edges(f'data/{data}/{behavior}.txt')
            for behavior in aux_behaviors
        }
        train_buy = load_edges(f'data/{data}/train.txt')
        test_buy = load_edges(f'data/{data}/test.txt')
        
        
        # Preprocess target-complemented behaviors
        train_buy_set = list2set(train_buy)
        not_buy_edges = {
            behavior: list2set(edges).difference(train_buy_set)
            for behavior, edges in behavior_edges.items()
        }
        
        for behavior, edges in not_buy_edges.items():
            save_edges(f'data/{data}/{behavior}_not_buy.txt', sorted(list(edges)))
        
        
        # Preprocess target-intersected behaviors
        buy_edges = {
            behavior: list2set(edges).intersection(train_buy_set)
            for behavior, edges in behavior_edges.items()
        }
        
        for behavior, edges in buy_edges.items():
            save_edges(f'data/{data}/{behavior}_buy.txt', sorted(list(edges)))
        

        for behavior, edges in behavior_edges.items():
            assert len(edges) == len(not_buy_edges[behavior]) + len(buy_edges[behavior])
        
        
        # Preprocess unified behavior graph
        all_edge = set(train_buy_set)
        for edges in behavior_edges.values():
            all_edge = all_edge.union(list2set(edges))
        all_edge = sorted(list(all_edge))
        save_edges(f'data/{data}/ubg.txt', all_edge)
        
        
        # Generate train/test buy interaction dict
        n_users = max(user for user, _ in all_edge)
        n_items = max(item for _, item in all_edge)
            
        train_dict = list2dict(train_buy, n_users)
        test_dict = list2dict(test_buy, n_items)
        
        with open(f'data/{data}/train.json', 'w') as f:
            json.dump(train_dict, f)
        with open(f'data/{data}/test.json', 'w') as f:
            json.dump(test_dict, f)
        
        
        # Generate data statistics
        bsg_types = aux_behaviors + ['buy']
        tcb_types = [f'{behavior}_not_buy' for behavior in aux_behaviors]
        tib_types = [f'{behavior}_buy' for behavior in aux_behaviors]
        trbg_types = tcb_types + tib_types
        
        data_statistics = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'bsg_types': bsg_types,
            'aux_types': aux_behaviors,
            'target_behavior': 'buy',
            'tcb_types': tcb_types,
            'tib_types': tib_types,
            'trbg_types': trbg_types,
            'n_view': len(behavior_edges['view']) if 'view' in behavior_edges else 0,
            'n_cart': len(behavior_edges['cart']) if 'cart' in behavior_edges else 0,
            'n_collect': len(behavior_edges['collect']) if 'collect' in behavior_edges else 0,
            'n_buy': len(train_buy),
            'n_view_not_buy': len(not_buy_edges['view']) if 'view' in not_buy_edges else 0,
            'n_cart_not_buy': len(not_buy_edges['cart']) if 'cart' in not_buy_edges else 0,
            'n_collect_not_buy': len(not_buy_edges['collect']) if 'collect' in not_buy_edges else 0,
            'n_view_buy': len(buy_edges['view']) if 'view' in buy_edges else 0,
            'n_cart_buy': len(buy_edges['cart']) if 'cart' in buy_edges else 0,
            'n_collect_buy': len(buy_edges['collect']) if 'collect' in buy_edges else 0,
            'n_ubg': len(all_edge)
        }
        with open(f'data/{data}/statistics.json', 'w') as f:
            json.dump(data_statistics, f)
                

if __name__ == '__main__':
    preprocess()
