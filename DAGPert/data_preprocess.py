import sys
import os
import pickle
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
import torch
import json
import pickle
import glob
import scipy.sparse as sp
from gears import PertData
from gears.utils import zip_data_download_wrapper, print_sys, filter_pert_in_go
from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter, rank_genes_groups_by_cov
from gears.utils import parse_single_pert, parse_combo_pert
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch


class CompressPertData(PertData):
    def __init__(self, data_path, 
                gene_set_path=None, 
                default_pert_graph=True):
        super().__init__(data_path, gene_set_path, default_pert_graph)

    def load(self, data_name = None, data_path = None, hvg = None, downsample=-1, set_flag=50, rewrite=False, pert_gene_list=[]):
        # 读取数据
        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path

            if hvg is not None:
                adata_path = os.path.join(data_path, f'perturb_processed_hvg{hvg}.h5ad')
                if not os.path.exists(adata_path):
                    adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            else:
                adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata_path = adata_path
        
            # adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            print(self.adata_path)
            self.adata = sc.read_h5ad(self.adata_path)
        elif os.path.exists(os.path.join(self.data_path, data_name)):
            data_path = os.path.join(self.data_path, data_name)
            if hvg is not None:
                adata_path = os.path.join(data_path, f'perturb_processed_hvg{hvg}.h5ad')
                if not os.path.exists(adata_path):
                    adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            else:
                adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata_path = adata_path
            self.adata = sc.read_h5ad(self.adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        ############ add the step to filter hvg
        if hvg is not None:
            pyg_path = os.path.join(data_path, f'hvg_{hvg}_data_pyg')
        else:
            pyg_path = os.path.join(data_path, 'data_pyg')

        if hvg is not None and 'hvg' not in self.adata_path:
            self.hvg = hvg
            sc.pp.highly_variable_genes(self.adata, flavor='cell_ranger')
            top_hvg = self.adata.var.nlargest(hvg, 'dispersions_norm')
            self.adata.var['highly_variable'] = self.adata.var['gene_name'].isin(top_hvg['gene_name'])

            pert_list = list(set(self.adata.obs['condition']))
            pert_gene_list += [item.split('+')[0] if '+' in item else item for item in pert_list] \
                + [item.split('+')[1] if '+' in item else item for item in pert_list] 
            pert_gene_list = list(set(pert_gene_list) - set(['ctrl'])) 
            print(len(pert_gene_list))
            self.adata.var['pert'] = self.adata.var['gene_name'].isin(pert_gene_list).astype(bool)
            self.adata = self.adata[:, self.adata.var['highly_variable'] | self.adata.var['pert']]
            duplicate_genes = self.adata.var['gene_name'].duplicated(keep=False)
            if duplicate_genes.sum() > 0:
                print(f"发现 {duplicate_genes.sum()} 个重复基因名称")
                # 可以选择只保留第一个出现的位置
                self.adata = self.adata[:, ~self.adata.var['gene_name'].duplicated()]
            print_sys(self.adata)
            self.adata = get_DE_genes(self.adata, False)
            self.adata = get_dropout_non_zero_genes(self.adata)
            
            self.adata_path = os.path.join(data_path, f'perturb_processed_hvg{hvg}.h5ad')
            print_sys("Saving hvg adata at" + self.adata_path)
            self.adata.write(self.adata_path)

        # 过滤不表达的perturbation基因
        print_sys('These perturbations are not in the expression and their '
                  'perturbation can thus not be predicted')
        not_in_exp_pert = np.array(self.adata.obs[
                            self.adata.obs.condition.apply(
                            lambda x:not filter_pert_in_go(x,
                                list(self.adata.var['gene_name'])))].condition.unique())
        print_sys(len(not_in_exp_pert))

        filter_exp = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, 
                                list(self.adata.var['gene_name'])))]
        self.adata = self.adata[filter_exp.index.values, :]
        self.set_pert_genes()

        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)

        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        self.set_pert_genes()

        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)

        if downsample > 0:
            print_sys(f"Downsampling the cells in every condition to {downsample} cells")
            sampled_indices = (
                self.adata.obs
                .loc[self.adata.obs['condition'] != 'ctrl']  # 过滤掉 ctrl
                .groupby('condition', group_keys=False)
                .apply(lambda x: x.sample(n=min(downsample, len(x)), random_state=42))
                .index
            )
            ctrl_indices = self.adata.obs[self.adata.obs['condition'] == 'ctrl'].index
            self.adata = self.adata[list(set(sampled_indices).union(set(ctrl_indices)).intersection(set(self.adata.obs_names)))]
            print(self.adata)
            pyg_path = os.path.join(pyg_path, f'downsample_{downsample}_set_{set_flag}')
        else:
            pyg_path = os.path.join(pyg_path, f'all_set_{set_flag}')

        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname) and not rewrite:
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            self.cell_nums = set_flag
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
    
    
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        de_idx: np.ndarray
            DE gene indices
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X)
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)
    

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        split_adata: anndata.AnnData
            Annotated data matrix
        pert_category: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []
        
        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            
            if self.cell_nums == 1:
                for i, cell_z in enumerate(adata_.X):
                    # Use samples from control as basal expression
                    ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                            len(self.ctrl_adata), num_samples), :]
                    for j, c in enumerate(ctrl_samples.X):
                        Xs.append(c)
                        ys.append(cell_z)

            else:
                for i in range(0, adata_.X.shape[0], self.cell_nums):
                    # 当前一批扰动细胞
                    pert_cells = adata_.X[i : i + self.cell_nums]
                    if pert_cells.ndim == 1:  # 保证pert_cells是两维的
                        pert_cells = pert_cells[np.newaxis, :]
                    while pert_cells.shape[0] < self.cell_nums:
                        pert_cells = pert_cells if isinstance(pert_cells, np.ndarray) else pert_cells.toarray()
                        temp = adata_.X[:self.cell_nums - pert_cells.shape[0]].toarray()
                        pert_cells = np.concatenate([pert_cells, temp], axis=0) # 需要对array形式进行处理

                    # 随机采样 control 数据
                    ctrl_samples = self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), self.cell_nums), :].X

                    Xs.append(ctrl_samples)
                    ys.append(pert_cells)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            if self.cell_nums == 1:
                for cell_z in adata_.X:
                    Xs.append(cell_z)
                    ys.append(cell_z)

            else:
                for i in range(0, adata_.X.shape[0], self.cell_nums):
                    temp = adata_.X[i : i + self.cell_nums]
                    Xs.append(temp)
                    ys.append(temp)
                    
        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            X = X if isinstance(X, np.ndarray) else X.toarray()
            y = y if isinstance(y, np.ndarray) else y.toarray()

            cell_graphs.append(self.create_cell_graph(X, y, de_idx, pert_category, pert_idx))

        return cell_graphs


class ChemPertData(CompressPertData):
    def __init__(self, data_path, 
                gene_set_path=None, 
                default_pert_graph=False):
        super().__init__(data_path, gene_set_path, default_pert_graph)

    def load(self, data_name = None, data_path = None, hvg = None, 
             downsample=-1, set_flag=50, rewrite=False, 
             condition_key = 'condition_name'):
        data_path = os.path.join(self.data_path, data_name)
        
        if hvg is not None:
            adata_path = os.path.join(data_path, f'perturb_processed_hvg{hvg}.h5ad')
            if not os.path.exists(adata_path):
                adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
        else:
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
        self.adata_path = adata_path
        self.adata = sc.read_h5ad(self.adata_path)
        self.dataset_name = data_name
        self.dataset_path = data_path

        if hvg is not None:
            pyg_path = os.path.join(data_path, f'hvg_{hvg}_data_pyg')
        else:
            pyg_path = os.path.join(data_path, 'data_pyg')

        if hvg is not None and 'hvg' not in self.adata_path:
            self.hvg = hvg
            if 'counts' not in self.adata.layers.keys():
                self.adata.layers['counts'] = self.adata.X.copy()
                sc.pp.normalize_per_cell(self.adata)
                sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, flavor='cell_ranger') # 高变化量基因
            top_hvg = self.adata.var.nlargest(hvg, 'dispersions_norm')
            self.adata.var['highly_variable'] = self.adata.var['gene_name'].isin(top_hvg['gene_name'])
            
            self.adata = self.adata[:, self.adata.var['highly_variable']]
            self.adata = get_DE_genes(self.adata, False)
            self.adata = get_dropout_non_zero_genes(self.adata)
            print_sys(self.adata)

            self.adata_path = os.path.join(data_path, f'perturb_processed_hvg{hvg}.h5ad')
            print_sys("Saving hvg adata at" + self.adata_path)
            self.adata.write(self.adata_path)

        self.set_pert_conds()

        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)

        if downsample > 0:
            print_sys(f"Downsampling the cells in every condition to {downsample} cells")
            
            sampled_indices = (
                self.adata.obs
                .loc[self.adata.obs['condition'] != 'ctrl']  # 过滤掉 ctrl
                .groupby(condition_key, group_keys=False)
                .apply(lambda x: x.sample(n=min(downsample, len(x)), random_state=42))
                .index
            )
            ctrl_indices = self.adata[self.adata.obs['condition'] == 'ctrl'].obs.index  
            self.adata = self.adata[list(set(sampled_indices).union(set(ctrl_indices)).intersection(set(self.adata.obs_names)))]
            print(self.adata)
            pyg_path = os.path.join(pyg_path, f'downsample_{downsample}_set_{set_flag}')
        else:
            pyg_path = os.path.join(pyg_path, f'all_set_{set_flag}')

        if not all(self.adata.obs['condition_name'].isin(self.adata.uns['rank_genes_groups_cov_all'].keys())):
            self.adata = get_DE_genes(self.adata, False)

        sampled_indices = self.adata[self.adata.obs['condition_name'].isin(self.adata.uns['rank_genes_groups_cov_all'].keys())].obs.index
        ctrl_indices = self.adata[self.adata.obs['condition'] == 'ctrl'].obs.index                                  
        adata = self.adata[list(set(sampled_indices).union(set(ctrl_indices)).intersection(set(self.adata.obs_names)))].copy()
        adata.uns = self.adata.uns.copy()
        self.adata = adata
        print(self.adata)
        
        if condition_key != 'condition':
            self.adata.obs['condition_backup'] = self.adata.obs['condition']
            self.adata.obs['condition'] = self.adata.obs['condition_name']
        
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname) and not rewrite:
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.gene_names = self.adata.var.gene_name
            self.cell_nums = set_flag
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")

    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Done!")


    def set_pert_conds(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """

        essential_condition = self.adata.obs['condition'].apply(lambda x:x.split('+')[0])

        # 可能会加有'ctrl'
        self.pert_names = np.unique(list(essential_condition))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        split_adata: anndata.AnnData
            Annotated data matrix
        pert_category: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if '_ctrl_' not in pert_category:
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition'][0]
            cell_type = adata_.obs['cell_type'][0] # 提取cell_type
            self.ctrl_adata = self.adata[self.adata.obs['condition'].str.contains(cell_type + '_ctrl', regex=False)]
            
            if self.ctrl_adata.shape[0] == 0:
                print_sys(f"Warning: No control cells found for cell type {cell_type}. \
                          Using all cells in the perturbed condition as control.")
                self.ctrl_adata = self.adata[self.adata.obs['condition'].str.contains('_ctrl', regex=False)]

            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes

            if self.cell_nums == 1:
                for i, cell_z in enumerate(adata_.X):
                    # Use samples from control as basal expression
                    ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                            len(self.ctrl_adata), num_samples), :]
                    for j, c in enumerate(ctrl_samples.X):
                        Xs.append(c)
                        ys.append(cell_z)

            else:
                for i in range(0, adata_.X.shape[0], self.cell_nums):
                    # 当前一批扰动细胞
                    pert_cells = adata_.X[i : i + self.cell_nums]
                    if pert_cells.ndim == 1:  # 保证pert_cells是两维的
                        pert_cells = pert_cells[np.newaxis, :]
                    while pert_cells.shape[0] < self.cell_nums:
                        pert_cells = pert_cells if isinstance(pert_cells, np.ndarray) else pert_cells.toarray()
                        temp = adata_.X[:self.cell_nums - pert_cells.shape[0]].toarray()
                        pert_cells = np.concatenate([pert_cells, temp], axis=0) # 需要对array形式进行处理

                    # 随机采样 control 数据
                    ctrl_samples = self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), self.cell_nums), :].X

                    Xs.append(ctrl_samples)
                    ys.append(pert_cells)
        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            if self.cell_nums == 1:
                for cell_z in adata_.X:
                    Xs.append(cell_z)
                    ys.append(cell_z)

            else:
                for i in range(0, adata_.X.shape[0], self.cell_nums):
                    temp = adata_.X[i : i + self.cell_nums]
                    Xs.append(temp)
                    ys.append(temp)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            X = X if isinstance(X, np.ndarray) else X.toarray()
            y = y if isinstance(y, np.ndarray) else y.toarray()

            cell_graphs.append(self.create_cell_graph(X, y, de_idx, 
                                                      pert_category, pert_idx))

        return cell_graphs

    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None,
                      test_cell_type=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}

        Returns
        -------
        None

        """
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom', 'cell_type']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        
        if split == 'custom':
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                    raise ValueError('Please set split_dict_path for custom split')
            return
            
        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        if split == 'cell_type':
            split_file = self.dataset_name + '_' + split + '_' + test_cell_type + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        else:
            split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                        +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print('here1')
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = ChemDataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
            elif split == 'cell_type':
                # cell type split
                if test_cell_type is None:
                    raise ValueError('Please provide test_cell_type for cell_type split')
                DS = ChemDataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_cell_type=test_cell_type,
                                      seed=seed)

            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = ChemDataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = ChemDataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = ChemDataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'
                 
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")


class ChemDataSplitter(DataSplitter):
    def __init__(self, adata, split_type='single', seen=0):
        super().__init__(adata, split_type, seen)

    def pert_split(self, pert):
        if len(pert.split('_')) == 3:
            pert = pert.split('_')[1]
        return pert
    
    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        perts = [self.pert_split(p) for p in perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)
    

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        def parse_any_pert(p):
            p = self.pert_split(p)
            if ('ctrl' in p) and (p != 'ctrl'):
                return [parse_single_pert(p)]
            elif 'ctrl' not in p:
                out = parse_combo_pert(p)
                return [out[0], out[1]]

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        return perts

    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)                
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in self.pert_split(x).split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)
        
        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in self.pert_split(x).split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
    

    def get_split_type(self, pert_list, test_cell_type):

        if test_cell_type is None:
            raise ValueError('Please provide test_cell_type for cell_type split')
        
        test_perts = [p for p in pert_list if test_cell_type in p]
        train_perts = [p for p in pert_list if (p not in test_perts)]
        return train_perts, test_perts    
    
    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')
        
        elif self.split_type == 'no_test':
            if test_perts is None:
                test_perts = np.random.choice(pert_list,
                                    int(len(pert_list) * test_size))
            

        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
                else:
                    test_perts = np.array(test_perts)
        else:
            if test_perts is None:
                test_perts = np.random.choice(combo_perts,
                                    int(len(combo_perts) * test_size))
                if len(combo_perts) == 0:
                    test_perts = np.random.choice(single_perts,
                                    int(len(single_perts) * test_size))

        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts
    

    def split_data(self, test_size=0.1, test_pert_genes=None,
                   test_perts=None, split_name='split', seed=None, val_size = 0.1,
                   train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75, only_test_set_perts = False,
                   test_cell_type=None):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if
                        '_ctrl_' not in p]# and '+ctrl' in p]
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('ctrl')
        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self.get_simulation_split_single(unique_perts,
                                                                  train_gene_set_size,
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)
        elif self.split_type == 'cell_type':
            train, test = self.get_split_type(unique_perts, test_cell_type)
            train, val = self.get_split_list(train, test_size=val_size) 
        elif self.split_type == 'no_test':
            train, val = self.get_split_list(unique_perts,
                                          test_size=val_size)      
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        if self.split_type != 'no_test':
            map_dict.update({x: 'test' for x in test})
        # map_dict.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata

def get_go_network(data_path, pert_list, data_name, k = 10):
    server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
    tar_data_download_wrapper(server_path, 
                            os.path.join(data_path, 'go_essential_all'),
                            data_path)
    df_jaccard = pd.read_csv(os.path.join(data_path, 
                                'go_essential_all/go_essential_all.csv'))

    df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                            ['importance'])).reset_index(drop = True)

    return df_out

def get_kg_disc(data_path):

    with open(data_path, 'r') as f:
        disc = json.load(f)

    return disc

def get_kg_graph(data_path=None):
    graph = None
    if data_path is not None:
        with open(data_path, 'rb') as f:
            graph = pickle.load(f)
    return graph

def generate_skeleton(gene_list):
    graph_dict = {}
    n = len(gene_list)
    for i, gene in enumerate(gene_list):
        target_vector = i  # 长度为 n
        rows = list(np.arange(n)) + [target_vector] * n
        cols = [target_vector] * n + list(np.arange(n))
        data = np.ones(2 * n)
        graph_dict[gene] = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    print('Generating skeleton finished! ')
    return graph_dict

def net_extract(net_dir, gene_index):
    csv_files = glob.glob(net_dir+"*.csv")

    df_list = []
    for netFile in csv_files:
        netDF = pd.read_csv(netFile)
        # print(netDF.head())
        netDF = netDF[(netDF.Gene1.isin(gene_index)) & (netDF.Gene2.isin(gene_index))]
        # Remove self-loops.
        netDF = netDF[netDF.Gene1 != netDF.Gene2]
        netDF.drop_duplicates(keep = 'first', inplace=True)
        df_list.append(netDF)

    netDF = pd.concat(df_list, ignore_index=True)
    netDF.drop_duplicates(keep = 'first', inplace=True)
    allNodes = set(netDF.Gene1.unique()).union(set(netDF.Gene2.unique()))
    nTFs = len(set(netDF.Gene1.unique()))
    nGenesinNet = len(allNodes)
    print("\n#TFs: %d, #Genes: %d, #Edges: %d, Density: %.3f" % (nTFs,nGenesinNet,netDF.shape[0],netDF.shape[0]/((nTFs*nGenesinNet)-nTFs)))
    
    G = nx.Graph()
    edges = [(row['Gene1'], row['Gene2']) for _, row in netDF.iterrows()]
    G.add_edges_from(edges)
    G.add_nodes_from(gene_index)
    
    print("the number of genes: ", len(set(gene_index)))
    print("the number of genes in graph: ", len(set(gene_index)))

    adj_matrix = nx.to_numpy_array(G, nodelist=gene_index, dtype=float)
    return torch.Tensor(adj_matrix)

def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        dataverse_download(url, save_path + '.tar.gz')
        print('Extracting tar file...')
        with tarfile.open(save_path  + '.tar.gz') as tar:
            tar.extractall(path= data_path)
        print("Done!")  

def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print('Found local copy...')
    else:
        print("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)        




