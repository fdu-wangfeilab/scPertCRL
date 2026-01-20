import sys
import os

sys.path.append('/home/dongjiaying/workbench/pert/scPertCRL/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from DAGPert.utils import *
from DAGPert.model.model_utils import *
from DAGPert.data_preprocess import *
from DAGPert.train_model_addDAGVAE import *
from DAGPert.eval import *
from omegaconf import OmegaConf

def construct_model(config, pert_data, device):
    # 3. load model 
    config.model.num_perts = len(pert_data.pert_names.tolist())
    edge_list = get_go_network(pert_data.data_path, pert_data.pert_names.tolist(), pert_data.dataset_name)
    sim_network = GeneSimNetwork(edge_list, pert_data.pert_names.tolist(), node_map=pert_data.node_map_pert)
    G_go = sim_network.edge_index
    G_go_weight = sim_network.edge_weight
    G_disc = get_kg_disc(config.data.desc_path) if config.data.desc_path != None and config.data.desc_path != "None" else None
    G_graph = None  # get_kg_graph(config.data.graph_path) if config.data.graph_path != 'None' else generate_skeleton(pert_data.adata.var['gene_name'])
    base_net = None # net_extract(config.data.network_path, pert_data.adata.var['gene_name'])
    gene_name = pert_data.adata.var['gene_name']
    
    config.model.input_dim = pert_data.adata.shape[1]
    if config.model.latent_dim is None:
        config.model.latent_dim = pert_data.adata.shape[1]
    print('latent_dim:', config.model.latent_dim)

    # 计算deg score
    rank_genes_groups = compute_degs(pert_data.adata)

    # 提取adata
    train_condition = pert_data.set2conditions['train']
    train_adata = pert_data.adata[pert_data.adata.obs['condition'].isin(train_condition)].copy()
    # print(train_adata)
    
    # 参数准备 
    config.model.dataset = config.data.dataset
    config.model.set_flag = config.data.set_flag

    # 模型构建
    model = Model(config.model, adata=train_adata,
                  task='gene_knockout', G_go=G_go, G_go_weight=G_go_weight, G_disc=G_disc,
                  G_graph=G_graph, base_net=base_net, gene_name=gene_name,
                  emb_flag=config.model.emb_flag, rank_genes_groups=rank_genes_groups, 
                  loss_type=config.model.loss_type, device=device)
    
    return model


def main(config, rep_flag=False):

    device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
    config.model.device = device
    save_path = os.path.join(config.output_path, config.data.dataset, "ablation", config.model.ablation_setting, str(config.seed))
    os.makedirs(save_path, exist_ok=True)
    set_random_seed(config.seed)

    # 2. load data
    pert_data = init_data(config)

    model = construct_model(config, pert_data, device)

    if config.exp.retrain_flag or not os.path.exists(save_path):
        model.train(pert_data.dataloader['train_loader'], 
            pert_data.dataloader['val_loader'], 
            config.train, seed=opt.seed, 
            output_path=save_path)
    else:
        model.load_model_dict(os.path.join(save_path, 'model'))


    test_res = model.eval_pert(pert_data.dataloader['test_loader'])

    compute_metrics_and_save(test_res = test_res, 
                             adata = pert_data.adata, 
                             output_path = save_path, 
                             rep = rep_flag)

    # 测试分类准确性
    model.eval_target(pert_data.dataloader['test_loader'])

    if config.exp.viz_flag:
        viz_and_save(pert_data, test_res, os.path.join(save_path, 'figure'), 
                    name=f'umap_{config.train.loss_weight}_{config.model.loss_type_weight}.pdf')
    
    if config.exp.save_data_flag:
        save_data(test_res, pert_data.adata, os.path.join(save_path, 'data'))
        print(test_res['pert_cat'].shape)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training')
    # 添加位置参数
    parser.add_argument('--opt_file', default="/home/dongjiaying/workbench/pert/project/config/knockout/scPertCRL.yaml", type=str, help='opt file')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--emb_flag', default='qwen', type=str)  
    parser.add_argument('--dataset', default='replogle_rpe1_essential', type=str, help='dataset name')
    parser.add_argument('--desc_path', default='/home/dongjiaying/workbench/pert/project/data/outputs/rpe1_input.json', type=str, help='desc path')
    parser.add_argument('--ab_set', default='all', type=str, help='the setting of ablation study.')
    parser.add_argument('--loss_type', default="combined4", type=str, help='hvg for training')
    parser.add_argument('--retrain_flag', action="store_true") 
    parser.add_argument('--scale', action="store_true") 
    parser.add_argument('--noise_scale', default=0.1, type=float)
    parser.add_argument('--net_type', default='cond', type=str)
    parser.add_argument('--num_pre_epochs', default=20, type=int)
    args = parser.parse_args()

    opt = OmegaConf.load(args.opt_file) # 加载配置文件

    opt.seed = args.seed
    opt.data.dataset = args.dataset
    opt.data.desc_path = args.desc_path

    opt.model.emb_flag = args.emb_flag
    opt.model.ablation_setting = args.ab_set
    opt.model.loss_type = args.loss_type
    
    opt.model.noise_scale = args.noise_scale
    opt.model.net_type = args.net_type
    opt.model.scale = args.scale
    opt.exp.retrain_flag = args.retrain_flag
    
    opt.train.num_pre_epochs = args.num_pre_epochs
    
    main(opt)
    