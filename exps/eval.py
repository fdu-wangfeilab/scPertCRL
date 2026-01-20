# %%
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import polars as pl
from cell_eval import MetricsEvaluator

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--path', type=str)

args = parser.parse_args()

path = f"{args.path}"
adata_ctrl = sc.read_h5ad(path + '/ctrl.h5ad')
adata_real = sc.read_h5ad(path + '/gt.h5ad')
adata_pred = sc.read_h5ad(path + '/pred.h5ad')

adata_real = sc.concat([adata_ctrl, adata_real])
adata_pred = sc.concat([adata_ctrl, adata_pred]) 

evaluator = MetricsEvaluator(
    adata_pred=adata_pred,
    adata_real=adata_real,
    control_pert="ctrl",
    pert_col="perturbation",
    num_threads=64,
    outdir = path, 
    prefix=f'CellEval',
)
(results, agg_results) = evaluator.compute()

cols = ['discrimination_score_l1', 'overlap_at_N', 'pearson_delta',  'mse', 'mae']
mean_row = agg_results.filter(pl.col("statistic") == "mean").select(cols)
print(mean_row)
