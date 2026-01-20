import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
import sys


sys.path.append('/home/dongjiayi/workbench/pert/scPertCRL/')
import matplotlib.pyplot as plt

import logging
from DAGPert.model.model_utils import *
from DAGPert.data_preprocess import *
from DAGPert.train_model_addDAGVAE import *
from DAGPert.eval import *

def compute_degs_with_cov(adata,
                        groupby,
                        control_group,
                        covariate,
                        pool_doses=False,
                        n_genes=50,
                        rankby_abs=True):

    
    gene_dict = []
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        gene_dict.append(adata_cov.uns['rank_genes_groups'])
    return gene_dict



def compute_degs(adata, mode='vscontrol', pval_threshold=0.05, condition_name='condition'):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.
    
    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)
    
    Returns:
        dict: rank_genes_groups results dictionary
        
    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == 'vsrest':
        # Remove control cells for vsrest analysis
        adata_subset = adata[adata.obs[condition_name] != 'ctrl'].copy()
        reference = 'rest'
    elif mode == 'vscontrol':
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = 'ctrl'
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # Compute DEGs
    sc.tl.rank_genes_groups(adata_subset, condition_name, method='t-test_overestim_var', reference=reference)
    
    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])
    
    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs[condition_name].unique(), desc=f"Computing DEGs {mode}"):
        if mode == 'vscontrol' and pert == 'ctrl':
            continue  # Skip control when comparing vs control
            
        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]
        
        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]
        
        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()
        
        deg_dict[pert] = {'up': pert_degs_sig_up, 'down': pert_degs_sig_down, 'logfc': pert_logfc}
    
    # Save results to adata.uns
    adata.uns[f'deg_dict_{mode}'] = deg_dict
    adata.uns[f'rank_genes_groups_{mode}'] = adata_subset.uns['rank_genes_groups'].copy()
    
    return adata_subset.uns['rank_genes_groups']



def init_chem_data(config):
    # 2. load data
    logging.info('loading the {} dataset'.format(config.data.dataset))
    pert_data = ChemPertData(config.data.data_root)  # specific saved folder
    pert_data.load(data_name=config.data.dataset, hvg=config.data.hvg, downsample=config.data.downsample, 
                   set_flag=config.data.set_flag, rewrite=False, condition_key='condition_name')  # specific dataset name # norman
    
    if config.data.dataset == "sci_plex_combo":
        pert_data.prepare_split(split='combo_seen2', combo_single_split_test_set_fraction=0.3, seed=config.seed)  # get data split with seed
        print("the number of condition:", len(pert_data.set2conditions['train']),  len(pert_data.set2conditions['val']), len(pert_data.set2conditions['test']))
    elif config.data.dataset == 'pbmc':
        pert_data.prepare_split(split='cell_type', test_cell_type=config.data.test_cell_type, seed=config.seed)
        print("the number of condition:", len(pert_data.set2conditions['train']),  len(pert_data.set2conditions['val']), len(pert_data.set2conditions['test']))

    pert_data.get_dataloader(batch_size=config.data.batch_size, test_batch_size=config.data.batch_size)
    print(pert_data.adata)
    return pert_data




def init_data(config):
    # 2. load data
    logging.info('loading the {} dataset'.format(config.data.dataset))
    pert_data = CompressPertData(config.data.data_root)  # specific saved folder
    pert_data.load(data_name=config.data.dataset, hvg=config.data.hvg, downsample=config.data.downsample, 
                   set_flag=config.data.set_flag, rewrite=False)  # specific dataset name # norman
    pert_data.prepare_split(split=config.data.split, seed=config.seed)  # get data split with seed
    pert_data.set2conditions['train'].remove('ctrl')
    pert_data.get_dataloader(batch_size=config.data.batch_size, test_batch_size=config.data.batch_size)
    print(pert_data.adata)
    return pert_data


