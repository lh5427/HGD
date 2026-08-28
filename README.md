# HGDM
This is the official code for **HGDM** (Hierarchical Graph Diffusion Model for Multi-Behavior Recommendation)



## Datasets
The statistics of datasets used in HGDM are summarized as follows.   
| Dataset | Users  | Items  | Views       | Collects        | Carts         | Buys   |
|---------|--------:|--------:|-------------:|-----------------:|---------------:|--------:|
| Taobao  | 15,449 | 11,953 | 873,954 | -          | 195,476  | 92,180 |
| Tmall   | 41,738 | 11,953 | 1,813,498 | 221,514   | 1,996    | 255,586|
| Beibei  | 21,716 | 7,977 | 2,412,586 | -         | 642,622   | 282,860|

<!--<img src="./assets/data_statistics.png" width="500px" height="200px" title="data statistics"/>-->


To preprocess the datasets for use in our code, type the following command:
```
python ./data/preprocess.py
```

## Usage
### Train our model from scratch
You can train the model with the best hyperparameters for each dataset by typing the following command in your terminal:

#### Train HGDM in the `Taobao` dataset
```python
python ./src/main.py --dataset taobao \
                     --aux_keep_ratio 0.8 \
                     --ubg_gnn_layers 3 \
                     --bsg_gnn_layers 1 \
                     --diff_gnn_layers 3 \
                     --lambda_md 0.001 \
                     --lambda_cl 0.05 \
                     --diff_loss_weight 0.01 \
                     --diff_sample_steps 20 \
                     --emb_dim 64 \
                     --num_epochs 100 \
                     --batch_size 1024
```

#### Train HGDM in the `Tmall` dataset
```python
python ./src/main.py --dataset tmall \
                     --aux_keep_ratio 0.8 \
                     --ubg_gnn_layers 4 \
                     --bsg_gnn_layers 1 \
                     --diff_gnn_layers 2 \
                     --lambda_md 0.2 \
                     --lambda_cl 0.05 \
                     --diff_loss_weight 0.1 \
                     --diff_sample_steps 200 \
                     --emb_dim 64 \
                     --num_epochs 100 \
                     --batch_size 1024
```

#### Train HGDM in the `Beibei` dataset
```python
python ./src/main.py --dataset beibei \
                     --aux_keep_ratio 0.4 \
                     --ubg_gnn_layers 3 \
                     --bsg_gnn_layers 4 \
                     --diff_gnn_layers 3 \
                     --lambda_md 0.05 \
                     --lambda_cl 0.05 \
                     --diff_loss_weight 0.05 \
                     --diff_sample_steps 200 \
                     --emb_dim 64 \
                     --num_epochs 100 \
                     --batch_size 1024
```

