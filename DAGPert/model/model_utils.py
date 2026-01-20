import numpy as np 
import random
import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import os
import math


def setup_seed(seed):
    """
    Set random seed.
    
    Parameters
    ----------
    seed
        Number to be set as random seed for reproducibility.
        
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    """Cite from https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_model_dict(path) # save best model here
        self.val_loss_min = val_loss


def kl_div(mu, var, prior_mu=None, prior_var=None):
    if prior_mu is None:
        return kl_divergence(Normal(mu, var.sqrt()),
                            Normal(torch.zeros_like(mu), torch.ones_like(var))).mean(dim=-1).mean()
    else:
        return kl_divergence(Normal(mu, var.sqrt()),
                            Normal(prior_mu, prior_var.sqrt())).mean(dim=-1).mean()

def l0_loss(
    p_m: torch.Tensor,
    p_v: torch.Tensor,
) -> torch.Tensor:
    l = (1 + torch.erf(((torch.abs(p_m)-0.1) / p_v) / math.sqrt(2))) / 2
    #l = (1 + torch.erf((torch.abs(q_m) / q_v) / math.sqrt(2))) / 2
    values, indices = torch.kthvalue(-l, 1, dim=-1)
    #l_ = (1 + torch.erf(((torch.abs(q_m)-0.1).sum(dim=0) / q_v.sum(dim=0)) * 5)) / 2
    return (l.mean(dim=-1) + 2 * values).mean()


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None,
    ) -> torch.Tensor:
    if mask==None:
        mask = torch.ones_like(input)
    else:
        mask = mask.float()
    per_element_loss = F.mse_loss(input, target, reduction="none")  # shape (B, D)
    masked_loss = per_element_loss * mask
    return masked_loss.sum() / mask.sum()


def wmse(x1, x2, weights):
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))



def annealed_weight(epoch, start=0, end=100, max_val=1.0, min_val=0.0):
    if epoch < start:
        return max_val
    elif epoch > end:
        return min_val
    else:
        return max_val - (max_val - min_val) * ((epoch - start) / (end - start))


from geomloss import SamplesLoss

MMDLoss = SamplesLoss(loss="gaussian", blur=0.05) 
EnergyLoss = SamplesLoss(loss="energy", blur=0.05)  
SinkhornLoss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)  


def combined_loss(target, pred, mask=None, mask_zero=False, sinkhorn_weight=0.001, energy_weight=1.0, mmd_weight=1000.0):
    b, s, _ = pred.shape
    if mask != None:
        pred = pred[mask].reshape(b, s, -1)
        target = target[mask].reshape(b, s, -1)
 
    sinkhorn_val = 0
    energy_val = 0
    mmd_val = 0
    for i in range(pred.shape[0]):
        if mask_zero and mask == None:
            nonzero_idx = (target[i].abs().sum(dim=0) > 10)
            pred_i = pred[i][:, nonzero_idx]
            target_i = target[i][:, nonzero_idx]
        else:
            pred_i = pred[i]
            target_i = target[i]
        
        assert not torch.isnan(pred_i).any()
        assert not torch.isnan(target_i).any()

        sinkhorn_val += SinkhornLoss(pred_i, target_i).nanmean()
        energy_val += EnergyLoss(pred_i, target_i).nanmean()
        mmd_val += MMDLoss(pred_i, target_i).nanmean()
    return (sinkhorn_weight * sinkhorn_val \
        + energy_weight * energy_val \
        + mmd_weight * mmd_val) / pred.shape[0]



def soft_histogram(x: torch.Tensor, bins: int = 50, min_val=None, max_val=None):
    """
    将连续数值转换为概率分布（soft histogram）
    x: [num_rows, num_cols]
    返回: [bins, num_cols] 概率分布
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    # 线性归一化到 [0, bins-1]
    scaled = (x - min_val) / (max_val - min_val) * (bins - 1)
    hist = F.one_hot(scaled.long().clamp(0, bins-1), num_classes=bins).float()
    hist = hist.sum(dim=0)  # 每列求和
    hist = hist / (hist.sum(dim=0, keepdim=True) + 1e-6) # 归一化为概率
    return hist

def kl_column_loss(pred: torch.Tensor, target: torch.Tensor, mask_zero: bool=True, bins: int = 50):
    loss = 0
    for i in range(pred.shape[0]):
        if mask_zero:
            nonzero_idx = (target[i].abs().sum(dim=0) > 10)
            pred_i = pred[i][:, nonzero_idx]
            target_i = target[i][:, nonzero_idx]
        else:
            pred_i = pred[i]
            target_i = target[i]
        pred_hist = soft_histogram(pred_i, bins)
        target_hist = soft_histogram(target_i, bins)
        eps = 1e-6
        kl = target_hist * torch.log((target_hist + eps) / (pred_hist + eps))
        kl_scaled = torch.log1p(kl)
        loss += kl_scaled.sum(dim=0).mean()
    return loss / pred.shape[0]


prompt_perturbation = f"""
[PERTURBATION TYPE]: {{pert}}
[GENES]: {{gene}}
[CELL TYPE]: {{cell}}
[KNOWLEDGE]: {{entries}}
"""

prompt_chem = f"""
[PERTURBATION TYPE]: {{pert}}
[COMPOUND]: {{chem}}
[CELL TYPE]: {{celltype}}
"""


cell_lines = {
    "K562": "K562 cells are immortalised myelogenous leukemia cells of the erythroleukemia type.",
    "A549": "A549 cells are a human lung cancer cell line.",
    "MCF7": "MCF7 cells are a human breast cancer cell line.",
    # pbmc
    "B cells": "B cells are lymphocytes that produce antibodies and mediate humoral immunity.",
    "Myeloid cells": "Myeloid cells are a diverse group of innate immune cells, including macrophages and neutrophils, involved in phagocytosis and inflammation.",
    "NK cells": "NK cells are natural killer cells that provide rapid defense against infected or cancerous cells without prior sensitization.",
    "T cells CD8+": "T cells CD8+ are cytotoxic T lymphocytes that directly kill infected or malignant cells.",
    "T cells CD4+": "T cells CD4+ are helper T cells that coordinate immune responses by activating other immune cells.",
    "T regulatory cells": "T regulatory cells are suppressor T cells that maintain immune tolerance and prevent autoimmune reactions.",
    # lincs
    "NPC": "NPC are neural progenitor cells derived from human pluripotent stem cells, used in neural development studies.",
    "HA1E": "HA1E are immortalized human kidney epithelial cells, commonly used in toxicology screening.",
    "BT20": "BT20 are human breast ductal carcinoma cells, representing triple-negative breast cancer.",
    "PC3": "PC3 are human prostate adenocarcinoma cells derived from bone metastasis, used in prostate cancer research.",
    "FIBRNPC": "FIBRNPC are fibroblast-derived neural progenitor cells, used in neural differentiation studies.",
    "HCC515": "HCC515 are human lung adenocarcinoma cells derived from a primary tumor.",
    "VCAP": "VCAP are human prostate carcinoma cells derived from vertebral metastasis.",
    "YAPC": "YAPC are human pancreatic adenocarcinoma cells with epithelial morphology.",
    "MCF7": "MCF7 are human breast adenocarcinoma cells, estrogen receptor-positive.",
    "A375": "A375 are human malignant melanoma cells derived from a solid tumor.",
    "A549": "A549 are human lung carcinoma cells with alveolar epithelial characteristics.",
    "MCF10A": "MCF10A are non-tumorigenic human mammary epithelial cells used as normal breast controls.",
    "HT29": "HT29 are human colorectal adenocarcinoma cells with epithelial morphology.",
    "CORL23": "CORL23 are human small cell lung carcinoma cells.",
    "SKBR3": "SKBR3 are human breast adenocarcinoma cells that overexpress HER2.",
    "MDAMB231": "MDAMB231 are human breast adenocarcinoma cells representing triple-negative breast cancer.",
    "SW620": "SW620 are human colorectal adenocarcinoma cells derived from lymph node metastasis.",
    "SKB": "SKB are human glioblastoma cells with stem-like properties.",
    "ASC.C": "ASC.C are adipose-derived stem cells from human subcutaneous fat.",
    "NOMO1": "NOMO1 are human acute monocytic leukemia cells.",
    "NPC.CAS9": "NPC.CAS9 are CRISPR/Cas9-engineered neural progenitor cells for genetic studies.",
    "SKL.C": "SKL.C are human acute myeloid leukemia cells with stem cell characteristics.",
    "HEPG2": "HEPG2 are human hepatocellular carcinoma cells with epithelial morphology.",
    "HEK293T": "HEK293T are human embryonic kidney cells expressing SV40 large T antigen for high transfection efficiency.",
    "HELA": "HELA are human cervical adenocarcinoma cells derived from Henrietta Lacks.",
    "HME1": "HME1 are human mammary epithelial cells immortalized with hTERT.",
    "SKL": "SKL are human acute myeloid leukemia cells derived from peripheral blood.",
    "THP1": "THP1 are human acute monocytic leukemia cells used for macrophage differentiation studies.",
    "MNEU.E": "MNEU.E are murine neural stem cells with epithelial characteristics.",
    "HUVEC": "HUVEC are human umbilical vein endothelial cells used in vascular biology.",
    "NCIH508": "NCIH508 are human colorectal adenocarcinoma cells with epithelial morphology.",
    "HS578T": "HS578T are human breast carcinoma cells derived from a carcinosarcoma.",
    "ASC": "ASC are adipose-derived mesenchymal stem cells from human adipose tissue.",
    "NKDBA": "NKDBA are natural killer cells derived from human peripheral blood.",
    "PHH": "PHH are primary human hepatocytes used in liver toxicity and metabolism studies.",
    "SW948": "SW948 are human colorectal adenocarcinoma cells derived from a primary tumor.",
    "NPC.TAK": "NPC.TAK are neural progenitor cells derived using the TAK differentiation protocol.",
    "TYKNU": "TYKNU are human ovarian clear cell carcinoma cells.",
    "SNUC4": "SNUC4 are human colorectal undifferentiated carcinoma cells.",
    "HUES3": "HUES3 are human embryonic stem cells used in pluripotency research.",
    "JHUEM2": "JHUEM2 are human endometrial cancer cells derived from a primary tumor.",
    "RMGI": "RMGI are human ovarian granulosa tumor cells.",
    "LOVO": "LOVO are human colorectal adenocarcinoma cells derived from metastatic site.",
    "HT115": "HT115 are human colorectal carcinoma cells used in RNAi studies.",
    "HUH7": "HUH7 are human hepatocellular carcinoma cells with well-differentiated features.",
    "DV90": "DV90 are human malignant pleural mesothelioma cells.",
    "U937": "U937 are human histiocytic lymphoma cells used in monocyte/macrophage studies.",
    "LNCAP": "LNCAP are human prostate adenocarcinoma cells expressing androgen receptor.",
    "NCIH1694": "NCIH1694 are human lung adenocarcinoma cells with epithelial characteristics.",
    "HCC15": "HCC15 are human squamous cell lung carcinoma cells.",
    "RKO": "RKO are human colon carcinoma cells deficient in DNA mismatch repair.",
    "H1299": "H1299 are human non-small cell lung carcinoma cells derived from lymph node metastasis.",
    "WSUDLCL2": "WSUDLCL2 are human diffuse large B-cell lymphoma cells.",
    "PL21": "PL21 are human acute promyelocytic leukemia cells with FLT3-ITD mutation.",
    "SKMEL1": "SKMEL1 are human malignant melanoma cells derived from metastatic site.",
    "U266": "U266 are human multiple myeloma cells producing immunoglobulin E.",
    "SKM1": "SKM1 are human acute myeloid leukemia cells with myelomonocytic features.",
    "SNU1040": "SNU1040 are human stomach adenocarcinoma cells.",
    "HL60": "HL60 are human promyelocytic leukemia cells used for neutrophil differentiation.",
    "A673": "A673 are human Ewing sarcoma cells with EWSR1-FLI1 fusion.",
    "CD34": "CD34 are human hematopoietic stem/progenitor cells isolated from cord blood.",
    "HEC108": "HEC108 are human endometrial carcinoma cells.",
    "JURKAT": "JURKAT are human acute T-cell leukemia cells used in immunology research.",
    "OV7": "OV7 are human ovarian carcinoma cells derived from ascites.",
    "MDST8": "MDST8 are human malignant peripheral nerve sheath tumor cells.",
    "EFO27": "EFO27 are human ovarian carcinoma cells derived from ascites fluid.",
    "NCIH596": "NCIH596 are human lung adenosquamous carcinoma cells.",
    "SKMEL28": "SKMEL28 are human malignant melanoma cells with epithelial morphology.",
    "SNUC5": "SNUC5 are human colorectal undifferentiated carcinoma cells.",
    "HCT116": "HCT116 are human colorectal carcinoma cells with microsatellite instability.",
    "SKLU1": "SKLU1 are human lung adenocarcinoma cells derived from pleural effusion.",
    "SNGM": "SNGM are human glioblastoma cells with stem-like properties.",
    "CL34": "CL34 are human chronic myeloid leukemia cells in blast crisis.",
    "AGS": "AGS are human gastric adenocarcinoma cells with epithelial morphology.",
    "RMUGS": "RMUGS are human ovarian mucinous cystadenocarcinoma cells.",
    "NCIH2073": "NCIH2073 are human lung adenocarcinoma cells with epithelial characteristics.",
    "COV644": "COV644 are human ovarian carcinoma cells derived from ascites.",
    "T3M10": "T3M10 are human pancreatic adenocarcinoma cells.",
    "NCIH1836": "NCIH1836 are human lung adenocarcinoma cells derived from a primary tumor.",
    "SW480": "SW480 are human colorectal adenocarcinoma cells derived from a primary tumor.",
    "HS27A": "HS27A are human foreskin fibroblasts used as feeder cells for stem cell culture.",
    "NEU": "NEU are neutrophils, the most abundant type of white blood cells in humans, accounting for 50-70% of circulating leukocytes. They provide rapid defense against bacterial and fungal infections through phagocytosis, degranulation, and neutrophil extracellular trap (NET) formation."
}



def set_random_seed(seed: int, deterministic: bool = True):
    """
    设置随机数种子以确保实验的可重复性。

    参数:
        seed (int): 随机数种子值。
        deterministic (bool): 是否启用确定性算法（可能会降低性能）。
    """
    # Python 内置随机数生成器
    random.seed(seed)
    
    # NumPy 随机数生成器
    np.random.seed(seed)
    
    # PyTorch 随机数生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 针对单个 GPU
    torch.cuda.manual_seed_all(seed)  # 针对多 GPU
    
    # 确保 CuDNN 使用确定性算法
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def get_pathway_mask(gmt_path, feature_list, max_pathway_dim):
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=300)
        mask,pathway = create_pathway_mask(feature_list=feature_list,
                                        dict_pathway=reactome_dict,
                                        add_missing=1,
                                        fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape)
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_pathway_dim, mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_pathway_dim, mask.shape[1]):])]

        return pathway, mask


def get_gmt(gmt):
    root = "/home/dongjiaying/workbench/pert/project/data/"
    gmt_files = {
        "human_gobp": [os.path.join(root, "resources/GO_bp.gmt")],
        "human_immune": [os.path.join(root, "resources/immune.gmt")],
        "human_reactome": [os.path.join(root, "resources/reactome.gmt")],
        "human_tf": [os.path.join(root, "resources/TF.gmt")],
        "mouse_gobp": [os.path.join(root, "resources/m_GO_bp.gmt")],
        "mouse_reactome": [os.path.join(root, "resources/m_reactome.gmt")],
        "mouse_tf": [os.path.join(root, "resources/m_TF.gmt")]
    }
    return gmt_files[gmt][0]


def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway


def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)


def negative_sampling_bce(logits, targets, neg_per_sample=100):
    """
    logits: (B, C)  未经过 sigmoid 的模型输出
    targets: (B, C) 0/1 标签
    neg_per_sample: 每个样本随机采多少个负类参与训练
    """
    B, C = logits.shape
    device = logits.device

    losses = []
    for i in range(B):
        pos_mask = targets[i] == 1
        neg_mask = targets[i] == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        # 如果没有正样本，就跳过或用全负样本
        if len(pos_indices) == 0:
            continue

        # 随机采样部分负类
        if len(neg_indices) > neg_per_sample:
            sampled_neg = neg_indices[logits[i, neg_indices].topk(neg_per_sample).indices]
        else:
            sampled_neg = neg_indices

        # 合并正负样本索引
        selected_indices = torch.cat([pos_indices, sampled_neg])

        # 取对应 logit 和 label
        sel_logits = logits[i, selected_indices]
        sel_targets = targets[i, selected_indices].float()

        # 计算 BCE loss
        loss = F.binary_cross_entropy_with_logits(sel_logits, sel_targets, reduction="mean")
        losses.append(loss)

    if len(losses) == 0:
        return F.binary_cross_entropy_with_logits(logits, targets)

    return torch.stack(losses).mean()


def fast_hard_negative_bce(logits, targets, neg_k=100):
    """
    每个样本取 top-k logits 最大的负类作为负样本。
    """
    B, C = logits.shape
    pos_mask = targets.bool()
    neg_mask = ~pos_mask

    # 屏蔽正样本的 logits
    masked_logits = logits.masked_fill(pos_mask, float('-inf'))

    # 每个样本 top-k 负类索引
    topk_vals, topk_idx = masked_logits.topk(neg_k, dim=1)

    # 构造 mask
    sampled_mask = torch.zeros_like(targets, dtype=torch.bool)
    sampled_mask.scatter_(1, topk_idx, True)

    final_mask = pos_mask | sampled_mask

    sel_logits = logits[final_mask]
    sel_targets = targets[final_mask].float()

    loss = F.binary_cross_entropy_with_logits(sel_logits, sel_targets, reduction="mean")
    return loss


def get_noise_scale(current_epoch, max_epochs, max_noise_scale):
    """
    根据当前 epoch 线性增加 noise_scale。
    
    参数:
        current_epoch (int): 当前 epoch（从 0 或 1 开始均可，但需一致）
        max_epochs (int): 总 epoch 数
        max_noise_scale (float): 最大 noise_scale
    
    返回:
        float: 当前 epoch 对应的 noise_scale
    """
    # 确保 current_epoch 不超过 max_epochs
    current_epoch = min(current_epoch, max_epochs)
    # 线性插值：从 0 到 max_noise_scale
    noise_scale = (current_epoch / max_epochs) * max_noise_scale
    return noise_scale