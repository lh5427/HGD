import os
import torch
from data import load_data
from model import HGDM, Trainer
from parser import parse_args
from utils import set_seed, print_args
from loguru import logger

def main(args):
    print_args(args)
    if args.seed is not None:
        set_seed(args.seed)
    data = load_data(
        args.data_dir, args.dataset, args.device, args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    if args.target_behavior is not None:
        data['target_behavior'] = args.target_behavior
    elif 'target_behavior' not in data:
        if 'buy' in data['bsg_types']:
            data['target_behavior'] = 'buy'
        else:
            data['target_behavior'] = data['bsg_types'][-1]

    model = HGDM(
        data,
        args.emb_dim,
        args.ubg_gnn_layers,
        args.bsg_gnn_layers,
        args.diff_gnn_layers,
        args.tda_layers,
        device=args.device,
        aux_sample_ratio=args.aux_sample_ratio,
        aux_keep_ratio=args.aux_keep_ratio,
        aux_keep_min=args.aux_keep_min,
        diff_loss_weight=args.diff_loss_weight,
        diff_sample_steps=args.diff_sample_steps,
        diff_noise_schedule=args.diff_noise_schedule,
        diff_noise_scale=args.diff_noise_scale,
        diff_noise_min=args.diff_noise_min,
        diff_noise_max=args.diff_noise_max,
        diff_history_num_per_term=args.diff_history_num_per_term,
        diff_beta_fixed=args.diff_beta_fixed,
        diff_infer_batch_size=args.diff_infer_batch_size,
        diff_time_emb_dim=args.diff_time_emb_dim,
        diff_dnn_dropout=args.diff_dnn_dropout,
        diff_dnn_norm=args.diff_dnn_norm,
        predict_item_chunk_size=getattr(args, 'predict_item_chunk_size', 256),
        use_momentum_distill=getattr(args, 'use_momentum_distill', True),
        momentum_tau=getattr(args, 'momentum_tau', 0.99),
        md_temperature=getattr(args, 'md_temperature', 0.2),
        lambda_md=getattr(args, 'lambda_md', 0.05),
        lambda_cl=getattr(args, 'lambda_cl', 0.1),
        cl_temperature=getattr(args, 'cl_temperature', 0.2),
    ).to(args.device)
    trainer = Trainer(model, data, args)

    if args.load_checkpoint:
        logger.info(f"Load checkpoint from {os.path.join(args.checkpoint_dir, args.dataset, 'model.pt')}")
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.dataset, 'model.pt'),
                                         map_location=args.device))
    else:
        logger.info("Start training the model")
        trainer.train_model()

    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Start evaluating the model")
    metrics = trainer.evaluate()
    logger.info(
        f"Test HR@10: {metrics['HR@10']:.4f}, "
        f"NDCG@10: {metrics['NDCG@10']:.4f}, "
        f"HR@20: {metrics['HR@20']:.4f}, "
        f"Recall@20: {metrics['Recall@20']:.4f}, "
        f"NDCG@20: {metrics['NDCG@20']:.4f}"
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
