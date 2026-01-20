import numpy as np

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance
import scipy
from geomloss import SamplesLoss
import torch
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from DAGPert.model.model_utils import masked_mse_loss, wmse


def generate_data_with_specific_num_of_DEG(test_res, adata, num_de_genes=50):
    
    if test_res['pred_de'].shape[-1] != num_de_genes:
        test_res_pred_de = np.zeros((len(test_res['pert_cat']), num_de_genes))
        test_res_ctrl_de = np.zeros((len(test_res['pert_cat']), num_de_genes))
        test_res_truth_de = np.zeros((len(test_res['pert_cat']), num_de_genes))
        
        # 创建数据
        print(f"generate data with top {num_de_genes} de genes.")
        for pert_category in np.unique(test_res['pert_cat']):
            adata_ = adata[adata.obs['condition'] == pert_category]
            if 'rank_genes_groups_cov_all' in adata_.uns:
                de_genes = adata_.uns['rank_genes_groups_cov_all']
                de = True
            else:
                de = False
                num_de_genes = 1  # 如果没有差异表达基因的记录，只取一个基因
            
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                # 获取对应的差异表达基因的索引
                de_idx = np.where(adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                # 如果没有差异表达基因信息，直接使用前 `num_de_genes` 个基因
                de_idx = np.arange(num_de_genes)

            # 创建新的测试数据，只保留指定数量的 DEG
            pred_de = test_res['pred'][test_res['pert_cat'] == pert_category][:, de_idx]
            truth_de = test_res['truth'][test_res['pert_cat'] == pert_category][:, de_idx]
            ctrl_de = test_res['ctrl'][test_res['pert_cat'] == pert_category][:, de_idx]

            # 根据需要生成其他部分的测试数据
            test_res_pred_de[test_res['pert_cat'] == pert_category] = pred_de
            test_res_truth_de[test_res['pert_cat'] == pert_category] = truth_de
            test_res_ctrl_de[test_res['pert_cat'] == pert_category] = ctrl_de

        test_res['pred_de'] = test_res_pred_de
        test_res['ctrl_de'] = test_res_ctrl_de
        test_res['truth_de'] = test_res_truth_de
    else:
        print("Directly use the predicted results.")

    return test_res

def compute_direction_of_change(x, y):

    pred_sign = np.sign(x)
    true_sign = np.sign(y)

    diff = np.abs(pred_sign - true_sign)

    total_genes = pred_sign.shape[0]
    frac_correct = np.sum(diff == 0) / total_genes
    frac_01 = np.sum(diff == 1) / total_genes
    frac_incorrect = np.sum(diff == 2) / total_genes

    return frac_correct, frac_01, frac_incorrect

def compute_direction_of_change_correct(pred, truth):
    res, _, _ = compute_direction_of_change(pred, truth)    
    return res

def compute_direction_of_change_01(pred, truth):
    _, frac_01, _ = compute_direction_of_change(pred, truth)
    return frac_01

def compute_direction_of_change_incorrect(pred, truth):     
    _, _, frac_incorrect = compute_direction_of_change(pred, truth)
    return frac_incorrect



def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'r2': r2_score,
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
        
        if len(results['pred'][p_idx].shape) == 3:
            pred = results['pred'][p_idx].mean(axis=(0, 1))
            truth = results['truth'][p_idx].mean(axis=(0, 1))
            pred_de = results['pred_de'][p_idx].mean(axis=(0, 1))
            truth_de = results['truth_de'][p_idx].mean(axis=(0, 1))
        else:
            pred = results['pred'][p_idx].mean(axis=0)
            truth = results['truth'][p_idx].mean(axis=0)
            pred_de = results['pred_de'][p_idx].mean(axis=0)
            truth_de = results['truth_de'][p_idx].mean(axis=0)
        
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(pred, truth)[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(pred, truth)

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'ctrl':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(pred_de, truth_de)[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(pred_de, truth_de)

                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert

def compute_ate_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
        #    'mse': mse,
           'r2': r2_score,
           'pearson': pearsonr,
           'frac_correct': compute_direction_of_change_correct,
           # 'frac_01': compute_direction_of_change_01,
           'frac_incorrect': compute_direction_of_change_incorrect,  
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):
        
        if pert == 'ctrl':
            continue

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]

        if len(results['pred'][p_idx].shape) == 3:
            pred = results['pred'][p_idx].mean(axis=(0, 1))
            truth = results['truth'][p_idx].mean(axis=(0, 1))
            ctrl = results['ctrl'][p_idx].mean(axis=(0, 1))
            pred_de = results['pred_de'][p_idx].mean(axis=(0, 1))
            truth_de = results['truth_de'][p_idx].mean(axis=(0, 1))
            ctrl_de = results['ctrl_de'][p_idx].mean(axis=(0, 1))
            pred_direction = pred - ctrl
            truth_direction = truth - ctrl
            pred_de_direction = pred_de - ctrl_de
            truth_de_direction = truth_de - ctrl_de
        else:
            pred = results['pred'][p_idx].mean(axis=0)
            truth = results['truth'][p_idx].mean(axis=0)
            ctrl = results['ctrl'][p_idx].mean(axis=0)
            pred_de = results['pred_de'][p_idx].mean(axis=0)
            truth_de = results['truth_de'][p_idx].mean(axis=0)
            ctrl_de = results['ctrl_de'][p_idx].mean(axis=0)
            pred_direction = pred - ctrl
            truth_direction = truth - ctrl
            pred_de_direction = pred_de - ctrl_de
            truth_de_direction = truth_de - ctrl_de

        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(pred_direction, truth_direction)[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(pred_direction, truth_direction)

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])


        if pert != 'ctrl':
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(pred_de_direction, truth_de_direction)[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(pred_de_direction, truth_de_direction)

                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])
        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0

    for m in metric2fct.keys():

        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])

    return metrics, metrics_pert

def compute_cls_metrics(results, adata, gt=None, cell_lines=None):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'auroc': roc_auc_score,
    }
    
    for m in metric2fct.keys():
        metrics[m+'_de'] = []
        metrics[m+'_dir'] = []

    for pert in np.unique(results['pert_cat']):
        
        if pert == 'ctrl':
            continue

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]

        y_truth = results['truth'][p_idx]
        y_pred = results['pred'][p_idx]
        y_ctrl = results['ctrl'][p_idx]

        for m, fct in metric2fct.items():
            if gt is not None:
                # 从表格中提取对应的perturb gene列表
                target_pert = pert.split('+')[0]
                genes = gt.loc[(gt['pert'] == target_pert) & (gt['label'] == 1), 'gene'].tolist()
                temp = np.isin(np.array(adata.var_names), genes).astype(int)
            else:
                if cell_lines is None:
                    genes = adata.uns['rank_genes_groups_cov_all'][pert][:100]
                else:
                    genes = adata.uns['rank_genes_groups_cov_all'][f'{cell_lines}_' + pert + '_1+1'][:100]
                temp = np.isin(np.array(adata.var_names), genes).astype(int)

            if sum(temp) == 0 or sum(temp) == len(temp):
                continue
            
            val = fct(y_true=temp, y_score=1/(1+np.exp(-(np.abs(y_pred.mean(0).mean(0) - y_ctrl.mean(0).mean(0))))) )

            metrics_pert[pert][m+ '_de'] = val
            metrics[m+ '_de'].append(metrics_pert[pert][m+ '_de'])

            t_dir = ((y_truth.mean(0).mean(0) - y_ctrl.mean(0).mean(0)) > 0).astype(float)
            if sum(t_dir) == 0 or sum(t_dir) == len(t_dir):
                continue

            p_dir = (y_pred.mean(0).mean(0) - y_ctrl.mean(0).mean(0)) * temp
            val = fct(y_true=t_dir, y_score=1/(1+np.exp(-(p_dir))))

            metrics_pert[pert][m+ '_dir'] = val
            metrics[m+ '_dir'].append(metrics_pert[pert][m+ '_dir'])
    
    for m in metric2fct.keys():
        metrics[m+ '_de'] = np.mean(metrics[m+ '_de'])
        metrics[m+ '_dir'] = np.mean(metrics[m+ '_dir'])
    return metrics, metrics_pert


def r2_score_on_deltas(delta_true, delta_pred, weights=None):
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan
    if np.any(np.isnan(delta_true)) or np.any(np.isnan(delta_pred)):
        return np.nan
    if weights is not None:
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        return r2_score(delta_true, delta_pred)


def gaussian_kernel(a, b, sigma=1.0):
    diff = np.expand_dims(a, 1) - np.expand_dims(b, 0)
    dist_sq = (diff ** 2).sum(axis=2)
    return np.exp(-dist_sq / (2 * sigma**2))

def compute_mmd(x, y, sigma=1.0):
    K_XX = gaussian_kernel(x, x, sigma)
    K_YY = gaussian_kernel(y, y, sigma)
    K_XY = gaussian_kernel(x, y, sigma)

    m = x.shape[0]
    n = y.shape[0]

    # 去掉对角线项，得到无偏估计
    k_xx = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
    k_yy = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
    k_xy = K_XY.mean()

    return k_xx + k_yy - 2 * k_xy

def compute_energy_distance(x, y, batch_size=32):
    assert x.shape[1] == y.shape[1], "Feature dimensions must match."

    n = min(len(x), len(y))
    if len(x) != len(y):
        # 随机下采样以匹配样本数
        idx = np.random.choice(len(x), n, replace=False)
        x = x[idx]
        idx = np.random.choice(len(y), n, replace=False)
        y = y[idx]

    losses = []
    num_batches = n // batch_size

    for i in range(num_batches):
        xb = x[i*batch_size:(i+1)*batch_size]
        yb = y[i*batch_size:(i+1)*batch_size]

        xb_perm = xb[np.random.permutation(batch_size)]
        yb_perm = yb[np.random.permutation(batch_size)]

        dXY = np.mean(np.sum((xb - yb)**2, axis=1))
        dXX = np.mean(np.sum((xb - xb_perm)**2, axis=1))
        dYY = np.mean(np.sum((yb - yb_perm)**2, axis=1))

        loss = 2 * dXY - dXX - dYY
        losses.append(loss)

    return np.mean(losses)


def compute_distribution_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """

    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mmd': SamplesLoss("gaussian", blur=0.05), # compute_mmd,
           'wasserstein': SamplesLoss(loss="sinkhorn", blur=0.05),
            'energy_distance': SamplesLoss("energy", blur=0.05), # compute_energy_distance, 
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):
        
        if pert == 'ctrl':
            continue

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
        

        pred = results['pred'][p_idx].reshape(-1, results['pred'][p_idx].shape[-1])
        truth = results['truth'][p_idx].reshape(-1, results['truth'][p_idx].shape[-1])
        pred_de = results['pred_de'][p_idx].reshape(-1, results['pred_de'][p_idx].shape[-1])
        truth_de = results['truth_de'][p_idx].reshape(-1, results['truth_de'][p_idx].shape[-1])

        for m, fct in metric2fct.items():
            val = fct(torch.from_numpy(pred).to(torch.float64), 
                        torch.from_numpy(truth).to(torch.float64))
            if np.isnan(val):
                val = 0

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

        for m, fct in metric2fct.items():
            if np.isnan(pred_de).any() or np.sum(pred_de) == 0:
                val = 0
            else:
                val = fct(torch.from_numpy(pred_de).to(torch.float64), torch.from_numpy(truth_de).to(torch.float64))
                if np.isnan(val):
                    val = 0
   
            metrics_pert[pert][m + '_de'] = val
            metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert


def compute_metrics_and_save(test_res, adata, output_path, rep=False):
    if rep == False:
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, 'eval_metrics.csv')
    else:
        csv_dir = os.path.join(output_path, 'replicate/')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f'eval_metrics_{rep}.csv')

    metrics = {
        "prediction": compute_metrics(test_res)[0],
        "distribution": compute_distribution_metrics(test_res)[0],
        "ATE": compute_ate_metrics(test_res)[0]
    }
    print(metrics)

    # ========= 3. 整理为 DataFrame =========
    rows = []
    for metric_type in metrics:
        if metrics[metric_type] == None: 
            continue
        for metric_name in metrics[metric_type]:
            rows.append({
                "Metric Type": metric_type,
                "Metric Name": metric_name,
                "Value": metrics[metric_type][metric_name],
            })

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False, sep='\t')  # 使用制表符分隔
    print(f"✅ 结果保存到: {csv_path}")




def save_data(test_res, adata, path):
    os.makedirs(path, exist_ok=True)
    
    if len(test_res["pred"].shape) == 3:
        pred = test_res["pred"].reshape(-1, test_res["pred"].shape[-1])
        truth = test_res["truth"].reshape(-1, test_res["truth"].shape[-1])
        ctrl = test_res['ctrl'].reshape(-1, test_res['ctrl'].shape[-1])
        pert_cat = np.tile(test_res["pert_cat"], (1, test_res["pred"].shape[1])).flatten()
    else:
        pred = test_res["pred"]
        truth = test_res["truth"]
        ctrl = test_res['ctrl']
        pert_cat = test_res["pert_cat"]

    test_adata = sc.AnnData(X=pred, 
                            obs={"perturbation": pert_cat},
                            var={"gene": adata.var_names.tolist()})

    test_adata.write_h5ad(f'{path}/pred.h5ad', compression='gzip')


    true_adata = sc.AnnData(X=truth, 
                        obs={"perturbation": pert_cat},
                        var={"gene": adata.var_names.tolist()})

    true_adata.write_h5ad(f'{path}/gt.h5ad', compression='gzip')

    ctrl_adata = sc.AnnData(X=ctrl, 
                        obs={"perturbation": ['ctrl'] * len(pert_cat)},
                        var={"gene": adata.var_names.tolist()})

    ctrl_adata.write_h5ad(f'{path}/ctrl.h5ad', compression='gzip')


def viz_and_save(pert_data, pred_res, output_path, name='umap.svg'):

    os.makedirs(output_path, exist_ok=True)
    sc.settings.figdir = output_path
    def umapWithPCA(traj_data, n_neighbors, min_dist, pca_pcs):
        from sklearn.decomposition import PCA
        from umap import UMAP
        pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
        umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_traj_data = umap_model.fit_transform(pca_model.fit_transform(traj_data))
        return umap_traj_data, umap_model, pca_model

    # 1. 可视化
    ctrl_X = pert_data.adata.X.toarray()
    
    if len(pred_res['truth'].shape) == 3:
        gt_y = pred_res['truth'].reshape(-1, pred_res['truth'].shape[-1])
        pred_y = pred_res['pred'].reshape(-1, pred_res['pred'].shape[-1])
        ctrl_x = pred_res['ctrl'].reshape(-1, pred_res['ctrl'].shape[-1])
    else:
        gt_y = pred_res['truth']
        pred_y = pred_res['pred']
        ctrl_x = pred_res['ctrl']

    pred_y[pred_y<0] = 0
    plt.figure(figsize=(10, 6))
    plt.hist(gt_y.flatten(), bins=50, edgecolor='black', alpha=0.7) 
    plt.hist((pred_y.flatten()), bins=50, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Matrix Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(output_path, 'hist'+name), bbox_inches='tight', dpi=100)

    all_data = np.vstack([gt_y, pred_y, ctrl_x])
    C_y = pred_res['pert_cat']
    adata_new = sc.AnnData(all_data)

    sc.tl.pca(adata_new, svd_solver='arpack')
    sc.pp.neighbors(adata_new, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata_new, min_dist=0.3) 

    label = ['Actual Cells' for c in range(len(C_y))] + ['Generated Cells' for _ in range(len(C_y))]\
    + ['Control Cells' for _ in range(len(C_y))]
    adata_new.obs['label'] = label

    sc.pl.umap(adata_new, 
            size=25, color=['label'], 
            groups=['Actual Cells', 'Generated Cells', 'Control Cells'], 
            legend_fontsize=14, title='All 14 Target Genes', 
            palette={'Actual Cells': 'blue', 
                        'Generated Cells': 'orange',
                        'Control Cells': 'grey',
                        'NA': 'grey'
                        },
            save=name,
            )

