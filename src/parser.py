import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='HGDM Settings')
    parser.add_argument('--dataset', type=str, default='taobao', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory of model checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for target data')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--ubg_gnn_layers', type=int, default=1,
                        help='Number of graph convolution layers for unified behavior graph learning')
    parser.add_argument('--bsg_gnn_layers', type=int, default=1,
                        help='Number of graph convolution layers for single-behavior graph learning')
    parser.add_argument('--diff_gnn_layers', type=int, default=1,
                        help='Number of graph convolution layers for denoised auxiliary behavior graph learning')
    parser.add_argument('--tda_layers', type=int, default=4, help='Number of tda layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:1', help='Training device')
    parser.add_argument('--topk', type=int, default=20, help='Top-k items')
    parser.add_argument('--eval_topks', type=int, nargs='+', default=[10, 20], help='Top-k values for evaluation metrics')

    parser.add_argument('--aux_sample_ratio', type=float, default=10,
                        help='Deprecated compatibility argument; current denoising uses aux_keep_ratio')
    parser.add_argument('--aux_keep_ratio', type=float, default=0.6,
                        help='Per-user ratio of original auxiliary edges kept in denoised auxiliary graphs')
    parser.add_argument('--aux_keep_min', type=int, default=1,
                        help='Minimum number of original auxiliary edges kept per user when available')
    parser.add_argument('--diff_loss_weight', type=float, default=0.1,
                        help='Weight of total diffusion loss')
    parser.add_argument('--diff_sample_steps', type=int, default=100,
                        help='Number of diffusion sampling steps')
    parser.add_argument('--diff_noise_schedule', type=str, default='linear-var',
                        choices=['linear', 'linear-var', 'cosine', 'binomial'],
                        help='Diffusion beta/noise schedule')
    parser.add_argument('--diff_noise_scale', type=float, default=0.1,
                        help='Diffusion noise scale')
    parser.add_argument('--diff_noise_min', type=float, default=0.0001,
                        help='Minimum diffusion noise value before scaling')
    parser.add_argument('--diff_noise_max', type=float, default=0.02,
                        help='Maximum diffusion noise value before scaling')
    parser.add_argument('--diff_history_num_per_term', type=int, default=10,
                        help='History length per timestep for diffusion importance sampling')
    parser.add_argument('--diff_beta_fixed', action=argparse.BooleanOptionalAction, default=True,
                        help='Fix the first diffusion beta to a small constant')
    parser.add_argument('--diff_infer_batch_size', type=int, default=256,
                        help='User chunk size for denoised auxiliary graph refresh')
    parser.add_argument('--diff_time_emb_dim', type=int, default=10,
                        help='Timestep embedding dimension for auxiliary diffusion DNN')
    parser.add_argument('--diff_dnn_dropout', type=float, default=0.5,
                        help='Dropout rate for auxiliary diffusion DNN')
    parser.add_argument('--diff_dnn_norm', action=argparse.BooleanOptionalAction, default=False,
                        help='Normalize diffusion DNN input interaction vectors')
    parser.add_argument('--predict_item_chunk_size', type=int, default=256,
                        help='Item chunk size for memory-efficient full-item scoring at evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Test batch size (smaller than train batch_size saves GPU memory at evaluation)')

    parser.add_argument('--target_behavior', type=str, default=None,
                        help='Target behavior for BPR supervision and score-fusion key behavior')

    parser.add_argument('--use_momentum_distill', action=argparse.BooleanOptionalAction, default=True,
                        help='Use momentum distillation between auxiliary and target behavior embeddings')
    parser.add_argument('--momentum_tau', type=float, default=0.99,
                        help='EMA momentum for the teacher graph encoder')
    parser.add_argument('--md_temperature', type=float, default=0.2,
                        help='Temperature for momentum distillation matching distributions')
    parser.add_argument('--lambda_md', type=float, default=0.05,
                        help='Loss weight for momentum distillation')

    parser.add_argument('--lambda_cl', type=float, default=0.001,
                        help='Loss weight for auxiliary-target contrastive learning (layer-2 and layer-3)')
    parser.add_argument('--cl_temperature', type=float, default=0.2,
                        help='Temperature for auxiliary-target InfoNCE contrastive loss')

    return parser.parse_args()
