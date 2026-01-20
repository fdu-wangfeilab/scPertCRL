import os
import random
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import copy

from tqdm import tqdm
from collections import defaultdict
from gears.inference import compute_metrics
from scipy.stats import pearsonr, spearmanr

from DAGPert.model.model_utils import *
from DAGPert.model.model_component import *
from DAGPert.model.model_ge import *
from torch_geometric.utils import to_dense_batch
from sklearn.preprocessing import MaxAbsScaler
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader

class Model(nn.Module):
    def __init__(self,       
        config,     
        task: str,   
        adata: sc.AnnData=None,
        G_go: torch.Tensor=None,
        G_go_weight: torch.Tensor=None,
        G_disc: dict=None,
        G_graph: dict=None, 
        gene_name: list=None,
        base_net: torch.Tensor=None,
        rank_genes_groups=None,
        emb_flag: str='all', 
        loss_type: str='combined',
        device: str='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        
        super(Model, self).__init__()
        
        self.config = config
        self.task = task
        self.emb_flag = emb_flag
        self.loss_type = loss_type
        self.device = device
        self.rank_genes_groups = rank_genes_groups
        self.task_learner = task_learner_selection(config, task, emb_flag, device,
                                                   G_disc, G_go, G_go_weight,)

        self.learner = DynamicDAGVAE(config, base_net, G_graph, gene_name) 

        if torch.cuda.is_available():
            self.task_learner = self.task_learner.cuda()
            self.learner = self.learner.cuda()
            self.device = torch.device('cuda')
        
        self.auroc = MultilabelAUROC(num_labels=config.input_dim, average="micro")
        self.ap = MultilabelAveragePrecision(num_labels=config.input_dim, average="micro")

        self.adata = adata
        self.is_train_finished = False
        self.gene_to_idx = {gene:id for id, gene in enumerate(adata.var_names.to_list())}

    def forward(self, batch_data, training_flag=True):
        task_emb = self.task_learner(batch_data, pred_flag=not training_flag)
        output = self.learner(batch_data, task_emb, training_flag=True)
        return output
    
    def set_train(self):
        self.task_learner.train()
        self.learner.train()
    
    def set_eval(self):
        self.task_learner.eval()
        self.learner.eval()
    
    def save_model_dict(self, output_path):
        if self.emb_flag in ['emb']:
            torch.save(self.task_learner.adaptor.state_dict(), output_path + '/adaptor.pt')
            torch.save(self.learner.state_dict(), output_path + '/model_dict.pt')
        else:
            torch.save(self.state_dict(), output_path + '/model_dict.pt')
    
    def load_model_dict(self, output_path):
        if self.emb_flag in ['emb']:
            self.task_learner.adaptor.load_state_dict(torch.load(os.path.join(output_path, 'adaptor.pt')))
            self.learner.load_state_dict(torch.load(os.path.join(output_path, 'model_dict.pt')))
        else:
            state_dict = torch.load(os.path.join(output_path, 'model_dict.pt'))
            self.load_state_dict(state_dict)

    def get_DE_genes(self, batch_data, fixed_de_idx=True, p_thresh=0.01):
        pert_flags = torch.zeros((len(batch_data.pert), batch_data.x.shape[-1]), dtype=torch.float32).to(self.device)            
        pert_vals = torch.zeros((len(batch_data.pert), batch_data.x.shape[-1]), dtype=torch.float32).to(self.device)      
        if fixed_de_idx:
            for i, des in enumerate(batch_data.de_idx):
                pert_flags[i, des] = 1
        else:
            for i, pert in enumerate(batch_data.pert):
                # 去掉 control 或空扰动（根据你的数据）
                if isinstance(self.rank_genes_groups, dict):
                    if pert == 'ctrl' or pert not in self.rank_genes_groups['names'].dtype.names:
                        continue
                    names = self.rank_genes_groups['names'][pert]          # shape: (n_top,)
                    pvals_adj = self.rank_genes_groups['pvals_adj'][pert]  # shape: (n_top,)
                    score = self.rank_genes_groups['scores'][pert]
                elif isinstance(self.rank_genes_groups, list):
                    for rg in self.rank_genes_groups:
                        if pert not in rg['names'].dtype.names:
                            continue
                        else:
                            names = rg['names'][pert]
                            pvals_adj = rg['pvals_adj'][pert]
                            score = rg['scores'][pert]
                            break

                sig_gene_names = [
                    name for name, p in zip(names, pvals_adj)
                    if p <= p_thresh
                ]

            
                # 将基因名转换为索引
                sig_indices = [self.gene_to_idx[gene] for gene in sig_gene_names if gene in self.gene_to_idx]
                
                # print(len(sig_indices))
                if len(sig_indices) > 0:
                    pert_flags[i, sig_indices] = 1.0
                else:
                    pert_flags[i, :] = 1.0

                temp = pvals_adj <= 0.01

                score_z = (score - score.min(axis=0)) / (score.max(axis=0) - score.min(axis=0) + 1e-6)
                pert_vals[:, temp] = torch.as_tensor(score_z, device=pert_vals.device)[temp] * 10
        #  print(pert_vals)
        return pert_flags


    def train(
        self,
        train_dataloader,
        val_dataloader,
        config, 
        seed: int = 19193,
        output_path = './', 
    ):

        self.is_train_finished = False

        if not seed is None:
            setup_seed(seed)
        
        low_decay_params = []
        high_decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'bn' in name:
                no_decay_params.append(param)
            elif 'cls' in name:
                high_decay_params.append(param)
            else:
                low_decay_params.append(param)

        self.optimizer = torch.optim.AdamW([{'params': low_decay_params, 'weight_decay': 1e-6}, 
                                           {'params': no_decay_params, 'weight_decay': 0.0},
                                           {'params': high_decay_params, 'weight_decay': 1e-4}], 
                                           lr=config.learning_rate * 2)

        self.early_pretrain_stopping = EarlyStopping(patience=config.patience, verbose=False)
        self.early_stopping = EarlyStopping(patience=config.patience, verbose=False)
        stop_flag = False

        # ---- Step 1: 计算 warmup 和总 epoch 数 ----
        num_epochs = config.num_epochs
        warmup_epochs = int(0.05 * num_epochs)  # 前5%

        # ---- Step 2: 定义两个调度器 ----
        # 1️⃣ Warmup阶段：学习率从 0 -> base_lr
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.3, end_factor=1.0, total_iters=warmup_epochs
        )

        # 2️⃣ Cosine阶段：从 base_lr 衰减到 eta_min
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
        )

        # ---- Step 3: 组合两个调度器 ----
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )

        if not os.path.exists(output_path + '/model'):
            os.makedirs(output_path + '/model', exist_ok=True)
        model_output_path = output_path + '/model'
        if not os.path.exists(output_path + '/pretrain_model'):
            os.makedirs(output_path + '/pretrain_model', exist_ok=True)
        pretrain_output_path = output_path + '/pretrain_model'

        print('warm up ...')
        with tqdm(total = config.num_pre_epochs, ncols=100) as pbar:
            pbar.set_description('warm up ...')
            for epoch in range(config.num_pre_epochs):
                self.set_train()

                epoch_loss = defaultdict(float)
                val_epoch_loss = defaultdict(float)
                
                for idx, batch in enumerate(train_dataloader): 
                    if torch.cuda.is_available():
                        batch = batch.cuda()

                    loss = self.cal_base_loss(batch, config.loss_weight, epoch=epoch)

                    self.optimizer.zero_grad()
                    sum(loss.values()).backward()
                    self.optimizer.step()

                    for k, v in loss.items():
                        epoch_loss[k] += loss[k].item()
                
                epoch_loss = {k:v/(idx+1) for k, v in epoch_loss.items()}        
                
                self.set_eval()
                for idx, batch in enumerate(val_dataloader):
                    
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    
                    loss = self.cal_base_loss(batch, config.loss_weight, epoch=epoch)

                    for k, v in loss.items():
                        val_epoch_loss[k] += loss[k].item()

                val_epoch_loss = {k:v/(idx+1) for k, v in val_epoch_loss.items()}

                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(sum(epoch_loss.values())), 
                    val='{:.4f}'.format(sum(val_epoch_loss.values())),
                    LR='{:.4f}'.format(self.optimizer.param_groups[0]['lr']),
                    )
            
                self.early_pretrain_stopping(sum(val_epoch_loss.values()), self, pretrain_output_path)
                if self.early_pretrain_stopping.early_stop:
                    print('Integrative training early stop, validation loss does not improve in ' + str(config.patience) + ' epoches!')
                    self.load_model_dict(pretrain_output_path)
                    break

        if config.num_pre_epochs > 0:
            self.learner.generator.base_net.requires_grad_(False)    
        elif config.num_pre_epochs == 0 and os.listdir(pretrain_output_path):
            print('Load pretrain model from ' + str(pretrain_output_path))
            self.load_model_dict(pretrain_output_path)
            self.learner.generator.base_net.requires_grad_(False)

        print('training ...')
        with tqdm(total = config.num_epochs, ncols=100) as pbar:
            pbar.set_description('training')
            for epoch in range(config.num_epochs):
                self.set_train()

                epoch_loss = defaultdict(float)
                val_epoch_loss = defaultdict(float)
                
                for idx, batch in enumerate(train_dataloader): 
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                
                    loss = self.cal_loss(batch, config.loss_weight, epoch=epoch)

                    self.optimizer.zero_grad()
                    sum(loss.values()).backward()
                    self.optimizer.step()

                    for k, v in loss.items():
                        epoch_loss[k] += loss[k].item()
                
                epoch_loss = {k:v/(idx+1) for k, v in epoch_loss.items()}        
                
                self.set_eval()
                for idx, batch in enumerate(val_dataloader):
                    
                    if torch.cuda.is_available():
                        batch = batch.cuda()
                    loss = self.cal_loss(batch, config.loss_weight, epoch=epoch)

                    for k, v in loss.items():
                        val_epoch_loss[k] += loss[k].item()

                val_epoch_loss = {k:v/(idx+1) for k, v in val_epoch_loss.items()}
       
                self.scheduler.step()
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(sum(epoch_loss.values())), 
                    val='{:.4f}'.format(sum(val_epoch_loss.values())),
                    LR='{:.4f}'.format(self.optimizer.param_groups[0]['lr']),
                    )
                
                if epoch % 5 == 0:
                    print(epoch_loss, val_epoch_loss)
                
                if (sum(val_epoch_loss.values()) > sum(epoch_loss.values())) or stop_flag:
                    stop_flag = True
                    self.early_stopping(sum(val_epoch_loss.values()), self, model_output_path)
                    if self.early_stopping.early_stop:
                        print('Integrative training early stop, validation loss does not improve in '+str(config.patience)+' epoches!')
                        self.load_model_dict(model_output_path)
                        break

        self.save_model_dict(model_output_path)
            
        self.is_train_finished = True
    

    def eval_pert(self, 
                dataloader):
        self.set_eval()

        pert_cat = []
        pred = []
        truth = []
        ctrl = []
        pred_de = []
        truth_de = []
        ctrl_de = []
        results = {}
        logvar = []

        for itr, batch in enumerate(dataloader):
            batch.to(self.device)
                
            pert_cat.extend(batch.pert)

            with torch.no_grad():
                x, _ = to_dense_batch(batch.x, batch.batch)
                y, _ = to_dense_batch(batch.y, batch.batch)
            
                task_emb = self.task_learner(batch, pred_flag=True)
                output = self.learner(batch, task_emb)
                recon, p, coff, z, (mu, var), (imu, ivar) = output
                if len(recon) == 3:
                    recon, _, _ = recon
                    p, _, _ = p

                pred.extend(p.cpu())
                truth.extend(y.cpu())
                ctrl.extend(x.cpu())

                # Differentially expressed genes
                for itr, de_idx in enumerate(batch.de_idx):
                    pred_de.append(p[itr, :, de_idx])
                    truth_de.append(y[itr, :, de_idx])
                    ctrl_de.append(x[itr, :, de_idx])

        # all genes
        results["pert_cat"] = np.repeat(np.array(pert_cat), pred[0].shape[0])
        pred = torch.stack(pred).reshape(-1, pred[0].shape[-1])
        truth = torch.stack(truth).reshape(-1, truth[0].shape[-1])
        ctrl = torch.stack(ctrl).reshape(-1, ctrl[0].shape[-1])
        results["pred"] = pred.detach().cpu().numpy().astype(float)
        results["truth"] = truth.detach().cpu().numpy().astype(float)
        results["ctrl"] = ctrl.detach().cpu().numpy().astype(float)

        pred_de = torch.stack(pred_de).reshape(-1, pred_de[0].shape[-1])
        truth_de = torch.stack(truth_de).reshape(-1, truth_de[0].shape[-1])
        ctrl_de = torch.stack(ctrl_de).reshape(-1, ctrl_de[0].shape[-1])
        results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
        results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)
        results["ctrl_de"] = ctrl_de.detach().cpu().numpy().astype(float)
        return results


    def eval_target(self, 
                dataloader):
        self.set_eval()
        pred = []
        truth = []
        pert_cat = []
        for itr, batch in enumerate(dataloader):
            batch.to(self.device)

            pert_cat.extend(batch.pert)

            with torch.no_grad():
                x, _ = to_dense_batch(batch.x, batch.batch)
                y, _ = to_dense_batch(batch.y, batch.batch)


                task_emb = self.task_learner(batch, pred_flag=False)
                recon, p, coff, z, (mu, var), (imu, ivar)= self.learner(batch, task_emb)

                if len(recon) == 3:
                    recon, _, _ = recon
                    p, _, _ = p
                base_coff, diff_coff = coff 

                target = self.learner.target_predict(noise=imu, weight=diff_coff-base_coff) 
                pert_flags = self.get_DE_genes(batch, fixed_de_idx=False)
            
                pred.extend(target.cpu())
                truth.extend(pert_flags.cpu())


        # all genes
        pert_cat = np.array(pert_cat)
        pred = torch.stack(pred).reshape(-1, pred[0].shape[-1])
        truth = torch.stack(truth).reshape(-1, truth[0].shape[-1])

        metrics = {}
        metrics_pert = {}
    
        auroc = []
        ap = []
        for pert in np.unique(pert_cat):
            metrics_pert[pert] = {}
            p_idx = np.where(pert_cat == pert)[0]
            sub_pred = pred[p_idx].mean(axis=(0)).reshape(1, -1)
            sub_truth = truth[p_idx].mean(axis=(0)).reshape(1, -1)

            auroc_score = self.auroc(sub_pred, sub_truth.to(torch.int32))
            ap_score = self.ap(sub_pred, sub_truth.to(torch.int32))

            metrics_pert[pert]['auroc'] = auroc_score.item()
            metrics_pert[pert]['ap'] = ap_score.item()

            auroc.append(auroc_score.item())
            ap.append(ap_score.item())
            
        auroc = [x for x in auroc if x != 0]
        ap = [x for x in ap if x != 0]
        print("AUROC:", np.mean(auroc))
        print("Average Precision:", np.mean(ap))

