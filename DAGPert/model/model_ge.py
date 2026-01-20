import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle

from torch_geometric.nn import SGConv
from torch.distributions import Normal
from torch.autograd import Variable
from DAGPert.model.model_utils import *
from DAGPert.model.model_base import *
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

import pickle as pkl
import pandas as pd
from sklearn.decomposition import PCA
import scipy.sparse as sparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import re


class PromptEmbeddingLearner(nn.Module):
    def __init__(self, config, G_disc):
        super(PromptEmbeddingLearner, self).__init__()

        self.device = config.device
        self.G_disc = G_disc
        self.config = config

        local_path = Path("~/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        "/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062/").expanduser()
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.prompt_model = AutoModelForMaskedLM.from_pretrained(local_path,
                            torch_dtype=torch.float16).to(self.device)
        config.prompt_embedding_dim = 768

        local_path = Path('/mnt/sda/zhouxiaochi/workbench/data_processing/data/gene_summary_embeddings_qwen.pickle')
        with open(local_path, "rb") as f:
            data = pickle.load(f)
        self.G_emb = data    
        config.token_embedding_dim = 2560

            # 冻结预训练模型参数
        for param in self.prompt_model.parameters():
            param.requires_grad = False
        
        config.embedding_dim = config.prompt_embedding_dim + config.token_embedding_dim 
        self.adaptor = NetBlock(
            nlayer=2,
            dim_list=[config.embedding_dim] + [config.hidden_dim] + [config.aux_dim],
            act_list=[nn.ReLU()] * 2,
            dropout_rate=0.3,
            norm_flag = 'ln',
        )
        # self.adaptor = nn.Linear(config.embedding_dim, config.aux_dim).to(self.model.device)

    def _gene_to_text(self, gene):
        """构建一个基因的自然语言描述"""
        if self.G_disc == None:
            return gene
        entries = self.G_disc.get(gene, gene) 
        if isinstance(entries, list):
            sampled = random.sample(entries, k=min(20, len(entries)))
            text_block = "\n".join(["- " + s for s in sampled])
        else:
            text_block = entries
        return text_block

    def _perturbation_to_text(self, gene_list):
        """组合多个基因的文本表示"""
        genes = ' + '.join([gene for gene in gene_list])
        entries = '\n'.join([self._gene_to_text(gene) for gene in gene_list if gene != 'ctrl'])
        return prompt_perturbation.format(
            pert='SINGLE-GENE' if len(gene_list) == 1 else 'MUlTI-GENE',
            gene=genes,
            entries=entries,
            cell='None'
        )

    def tokenize(self, pert_texts):
         # tokenizer 编码
        tokenized = self.prompt_tokenizer(
            pert_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # forward 生成 embedding
        with torch.no_grad():
            outputs = self.prompt_model(**tokenized, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            cls_embedding = hidden[:, 0]   
        return cls_embedding
    
    def _gene_to_emb(self, gene):
        """构建一个基因的嵌入向量"""
        if self.G_emb == None:
            raise ValueError("Need to define model name in qwen.")
        emb = self.G_emb.get(gene) 
        return torch.Tensor(emb).to(self.device) if emb is not None else torch.zeros(self.embedding_dim).to(self.device)


        
    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        # 使用缓存数据，避免实时采样
        # 支持 str 或 list 的 batch_data.pert 表达形式
        if isinstance(batch_data.pert[0], str):
            batch_gene_lists = [[g for g in p.split('+') if g != 'ctrl'] for p in batch_data.pert]
        elif isinstance(batch_data.pert[0], list):
            batch_gene_lists = [[g for g in p if g != 'ctrl'] for p in batch_data.pert]
        else:
            raise ValueError("Unsupported format in batch_data.pert")
        
        prompt_texts = [self._perturbation_to_text(gene_list) for gene_list in batch_gene_lists]
        gene_list =  [gene for gene_list in batch_gene_lists for gene in gene_list]
        index_list = [idx for i, gene_list in enumerate(batch_gene_lists) for idx in len(gene_list) * [i]]

        prompt_emb = self.tokenize(prompt_texts)
        pert_emb = [self._gene_to_emb(gene) for gene in gene_list]
        temp = torch.stack(pert_emb)
        pert_emb = torch.zeros(len(batch_data.pert), temp.shape[1], dtype=temp.dtype).to(temp.device)
        expanded_indices = torch.LongTensor(index_list).view(-1, 1).repeat(1, temp.shape[1]).to(temp.device)
        pert_emb.scatter_add_(dim=0, index=expanded_indices, src=temp)

        cls_embedding = torch.cat([prompt_emb, pert_emb], dim=-1)

        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > 0).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()

        return self.adaptor(cls_embedding.float())


class PromptTokenLearner(nn.Module):
    def __init__(self, config, fusion='crossattn'):
        super(PromptTokenLearner, self).__init__()
        
        # 确定设备
        self.device = config.device
        self.config = config 
        local_path = Path("~/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca/").expanduser()
        self.token_tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.token_model = AutoModel.from_pretrained(local_path)
        config.token_embedding_dim = 384  # ChemBERTa的隐藏层维度为384

        local_path = Path("~/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062/").expanduser()
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.prompt_model = AutoModel.from_pretrained(local_path)
        config.prompt_embedding_dim = 768  # BiomedBERT的隐藏层维度为768
        
        # 冻结预训练模型参数
        for param in self.token_model.parameters():
            param.requires_grad = False
        for param in self.prompt_model.parameters():
            param.requires_grad = False

        self.fusion = fusion
        if self.fusion == 'gate':
            self.proj = nn.Linear(config.token_embedding_dim + config.prompt_embedding_dim, config.hidden_dim)

            self.gate = nn.Sequential(
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )

            self.chem_proj = nn.Linear(config.token_embedding_dim, config.hidden_dim)
            self.prompt_proj = nn.Linear(config.prompt_embedding_dim, config.hidden_dim)
        elif self.fusion == 'crossattn':
            self.chem_proj = nn.Linear(config.token_embedding_dim, config.hidden_dim)
            self.prompt_proj_k = nn.Linear(config.prompt_embedding_dim, config.hidden_dim)
            self.prompt_proj_v = nn.Linear(config.prompt_embedding_dim, config.hidden_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim=config.hidden_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.prompt_proj = nn.Linear(config.prompt_embedding_dim, config.aux_dim)
            self.token_proj = nn.Linear(config.token_embedding_dim, config.hidden_dim)
        

        self.adaptor = NetBlock(
            nlayer=2,
            dim_list= [config.hidden_dim] * 2 + [config.aux_dim],
            act_list=[nn.ReLU()] * 2,
            dropout_rate=0.5,
            norm_flag = 'ln',
        )
    
    def pre_forward(self, batch_data, pred_flag=True):
        prompt_texts = [cell_lines[pert.split('_')[0]] for pert in batch_data.pert]
        prompt_inputs = self.prompt_tokenizer(prompt_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        prompt_outputs = self.prompt_model(**prompt_inputs)
        prompt_emb = prompt_outputs.last_hidden_state[:, 0, :]
        prompt_emb = self.prompt_proj(prompt_emb)
        return prompt_emb

    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        # 处理输入数据
        prompt_texts = [cell_lines[pert.split('_')[0]] for pert in batch_data.pert]  # 移除后缀

        chem_texts = [p.split('_')[1] for p in batch_data.pert]
        batch_chem_texts = [[g for g in p.split('+') if g != 'ctrl'] for p in chem_texts]

        chem_list =  [chem for chem_list in batch_chem_texts for chem in chem_list]
        index_list = [i for i, chems in enumerate(batch_chem_texts) for _ in chems]

        prompt_inputs = self.prompt_tokenizer(prompt_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        chem_inputs = self.token_tokenizer(chem_list, padding=True, truncation=True, return_tensors="pt").to(self.device)

        prompt_outputs = self.prompt_model(**prompt_inputs)
        chem_outputs = self.token_model(**chem_inputs)

        chem_emb = chem_outputs.last_hidden_state[:, 0, :]
        chem_emb_agg = torch.zeros(len(batch_chem_texts), chem_emb.shape[1]).to(self.device)
        index_tensor = torch.tensor(index_list, device=self.device).unsqueeze(1).repeat(1, chem_emb.shape[1])
        chem_emb_agg.scatter_add_(dim=0, index=index_tensor, src=chem_emb)
        cls_embedding = self.token_proj(chem_emb_agg)

        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > 0).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()

        return self.adaptor(cls_embedding)


class SmileLearner(nn.Module):
    def __init__(self, config):
        super(SmileLearner, self).__init__()
        self.device = config.device
        self.embedding_dim = 1024
        self.config = config
        self.adaptor = nn.Linear(self.embedding_dim, config.aux_dim)
    
    def forward(self, batch, pred_flag=True, noise_scale=None):
        drug_len = len(batch.pert)
        fcfp4_array = np.zeros((drug_len, self.embedding_dim))

        pert_ids = [pert.split('_')[1] for pert in batch.pert]  # 移除后缀
        dose_ids = [pert.split('_')[2] for pert in batch.pert]
        generator = GetMorganGenerator(radius=2, fpSize=self.embedding_dim)
        for i, smiles in enumerate(pert_ids):
            smiles_list = re.split(r'\+(?!\])', smiles) # smiles.split('+')
            dose_list = dose_ids[i].split('+')
            for j, smi in enumerate(smiles_list):
                if smi == 'ctrl':
                    continue
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = generator.GetFingerprint(mol).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*float(dose_list[j])
                fcfp4_array[i] += fcfp4_list
        
        cls_embedding = torch.tensor(fcfp4_array).to(self.device)


        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > 0).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()
        return self.adaptor(cls_embedding)


class ChemLearner(nn.Module):
    def __init__(self, config, model_name='ChemBERT', fusion='late'):
        super(ChemLearner, self).__init__()
        
        # 确定设备
        self.device = config.device
        self.config = config 
        self.fusion = fusion
        if model_name == 'ChemBERT':
            local_path = Path("~/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca/").expanduser()
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModel.from_pretrained(local_path).to(self.device)
            config.embedding_dim = 384  # ChemBERTa的隐藏层维度为384
            
        elif model_name == 'BiomedBERT':
            local_path = Path("~/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062/").expanduser()
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModel.from_pretrained(local_path).to(self.device)
            config.embedding_dim = 768  # BiomedBERT的隐藏层维度为768
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 冻结预训练模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 适配器层
        self.adaptor = nn.Linear(config.embedding_dim, config.aux_dim).to(self.device)

    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        if self.fusion == 'early':
            # 处理输入数据
            pert_ids = [prompt_chem.format(
                        pert="SINGLE COMPOUND" if len(pert.split('_')[1].split('+')) == 1 else 'MUlTI-GENE',
                        celltype=cell_lines[pert.split('_')[0]],  # 获取pert的基因部分
                        chem=pert.split('_')[1]
                    ) for pert in batch_data.pert]  # 移除后缀

                    
            inputs = self.tokenizer(
                pert_ids, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # 将输入移至模型所在设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = self.model(**inputs)
            
            # 获取[CLS] token的嵌入（索引0）
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        elif self.fusion == 'late':
            chem_texts = [p.split('_')[1] for p in batch_data.pert]
            batch_chem_texts = [[g for g in p.split('+') if g != 'ctrl'] for p in chem_texts]

            chem_list =  [chem for chem_list in batch_chem_texts for chem in chem_list]
            index_list = [idx for i, chem_list in enumerate(batch_chem_texts) for idx in len(chem_list) * [i]]

            chem_inputs = self.tokenizer(chem_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
            chem_outputs = self.model(**chem_inputs)
            chem_emb = chem_outputs.last_hidden_state[:, 0, :]

            cls_embedding = torch.zeros(len(batch_chem_texts), chem_emb.shape[1]).to(self.device)
            index_tensor = torch.tensor(index_list, device=self.device).unsqueeze(1).repeat(1, chem_emb.shape[1])
            cls_embedding.scatter_add_(dim=0, index=index_tensor, src=chem_emb)
        
        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > 0).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()

        # 通过适配器层
        return self.adaptor(cls_embedding)


class CondLearner(nn.Module):
    def __init__(self, config):
        super(CondLearner, self).__init__()

        self.model = nn.Embedding(num_embeddings=config.num_perts, embedding_dim=config.hidden_dim)

    def forward(self, batch_data):
        batch_size = len(batch_data.pert_idx)

        # 获取设备
        device = batch_data.x.device if hasattr(batch_data, "x") and isinstance(batch_data.x, torch.Tensor) else "cpu"

        # 转换 pert_idx 为 long 类型，并放到正确的设备
        pert_idx = torch.as_tensor(batch_data.pert_idx, dtype=torch.long, device=device)
        pert_idx[pert_idx == -1] = 0
        # print(pert_idx.max(), pert_idx.min())  # 确保索引值在 (0, num_embeddings-1) 之间
        # 确保形状正确
        pert_idx = pert_idx.contiguous().view(batch_size, -1)

        # 计算 embedding
        embeddings = self.model(pert_idx)  # 形状: [batch_size, seq_len, hidden_dim]

        return embeddings.reshape(batch_size, -1)  # 变为 [batch_size, seq_len * hidden_dim]
    

class OnehotLearner(nn.Module):
    def __init__(self, config):
        super(OnehotLearner, self).__init__()

        self.pert_dim = config.num_perts
        self.model = nn.Sequential(
           nn.Linear(config.num_perts, config.aux_dim)
        ) 

    def get_pert_idx(self, batch):
        pert_num = len(batch.pert_idx)
        max_len = max(len(item) for item in batch.pert_idx)

        # 直接创建 long 类型的张量
        pert_idx = torch.full((pert_num, max_len), -1, dtype=torch.long, device=batch.x.device)
        
        for i, item in enumerate(batch.pert_idx):
            pert_idx[i, :len(item)] = torch.tensor(item, dtype=torch.long, device=batch.x.device)
        
        pert_idx[pert_idx == -1] = 0
        
        pert_onehot = torch.zeros(pert_num, self.pert_dim, device=batch.x.device)
        pert_onehot.scatter_(1, pert_idx, 1)
        pert_onehot = pert_onehot.clamp(0, 1)

        return pert_onehot.float()


    def forward(self, batch_data):
        pert_idx = self.get_pert_idx(batch_data)

        # 计算 embedding
        embeddings = self.model(pert_idx) 

        return embeddings
    

class FreezedEmbeddingLearner(nn.Module):
    def __init__(self, config,  device, model_name='qwen', fusion='late'):
        super(FreezedEmbeddingLearner, self).__init__()

        self.device = device
        self.model_name = model_name
        self.fusion = fusion
        self.config = config

        if model_name == 'qwen':
            local_path = Path('/mnt/sda/zhouxiaochi/workbench/data_processing/data/gene_summary_embeddings_qwen.pickle')
            with open(local_path, "rb") as f:
                data = pickle.load(f)
            self.G_emb = data    
            config.embedding_dim = 2560
            self.embedding_dim = config.embedding_dim
        elif model_name == 'ESM2':
            local_path = Path('/mnt/sda/zhouxiaochi/workbench/data_processing/data/ESM2_pert_features.pickle')
            with open(local_path, "rb") as f:
                data = pickle.load(f)
            self.G_emb = data    
            config.embedding_dim = 5120     
            self.embedding_dim = config.embedding_dim
        elif model_name == 'GENEPT':
            local_path = Path('/mnt/sda/zhouxiaochi/workbench/data_processing/data/GenePT_gene_protein_embedding_model_3_text.pickle')
            with open(local_path, "rb") as f:
                data = pickle.load(f)
            self.G_emb = data    
            config.embedding_dim = 3072     
            self.embedding_dim = config.embedding_dim


        self.adaptor = nn.Sequential( # nn.Dropout(self.config.noise_scale), 
            NetBlock(
            nlayer=2,
            dim_list=[config.embedding_dim] + [config.hidden_dim] + [config.aux_dim],
            act_list=[nn.ReLU()] * 2,
            dropout_rate=0.3,
            norm_flag = 'ln',
        ))
        # self.adaptor = nn.Linear(config.embedding_dim, config.aux_dim).to(self.model.device)

    def _gene_to_emb(self, gene):
        """构建一个基因的嵌入向量"""
        if self.G_emb == None:
            raise ValueError("Need to define model name in qwen.")
        emb = self.G_emb.get(gene) 
        return torch.Tensor(emb).to(self.device) if emb is not None else torch.zeros(self.embedding_dim).to(self.device)

    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        # 使用缓存数据，避免实时采样
        # 支持 str 或 list 的 batch_data.pert 表达形式
        if isinstance(batch_data.pert[0], str):
            batch_gene_lists = [[g for g in p.split('+') if g != 'ctrl'] for p in batch_data.pert]
        elif isinstance(batch_data.pert[0], list):
            batch_gene_lists = [[g for g in p if g != 'ctrl'] for p in batch_data.pert]
        else:
            raise ValueError("Unsupported format in batch_data.pert")
        gene_list =  [gene for gene_list in batch_gene_lists for gene in gene_list]
        index_list = [idx for i, gene_list in enumerate(batch_gene_lists) for idx in len(gene_list) * [i]]

        pert_embs = [self._gene_to_emb(gene) for gene in gene_list]
        temp = torch.stack(pert_embs)
        cls_embedding = torch.zeros(len(batch_data.pert), temp.shape[1]).to(self.device)
        expanded_indices = torch.LongTensor(index_list).view(-1, 1).repeat(1, temp.shape[1]).to(self.device)
        cls_embedding.scatter_add_(dim=0, index=expanded_indices, src=temp)

        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > self.config.noise_scale).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()
        return self.adaptor(cls_embedding)

    
class EmbeddingLearner(nn.Module):
    def __init__(self, config, G_disc, device, model_name='BiomedBERT', freeze_model=True, fusion='late'):
        super(EmbeddingLearner, self).__init__()

        self.device = device
        self.model_name = model_name
        self.G_disc = G_disc
        self.freeze_model = freeze_model
        self.fusion = fusion
        self.config = config

        if model_name == 'BiomedBERT':
            local_path = Path("~/.cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
            "/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062/").expanduser()
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModelForMaskedLM.from_pretrained(local_path,
                                torch_dtype=torch.float16).to(self.device)
            config.embedding_dim = 768
        else:
            # 直接加载 tokenizer 和模型
            self.tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
            self.model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral", 
                                                torch_dtype=torch.float16,
                                                attn_implementation="flash_attention_2").to(self.device) # 启用flash attention
            config.embedding_dim = 4096

        if freeze_model:
            # 冻结预训练模型参数
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.adaptor = NetBlock(
            nlayer=2,
            dim_list=[config.embedding_dim] + [config.hidden_dim] + [config.aux_dim],
            act_list=[nn.ReLU()] * 2,
            dropout_rate=0.3,
            norm_flag = 'ln',
        )
        # self.adaptor = nn.Linear(config.embedding_dim, config.aux_dim).to(self.model.device)

    def _gene_to_text(self, gene):
        """构建一个基因的自然语言描述"""
        if self.G_disc == None:
            return gene
        entries = self.G_disc.get(gene, gene) 
        if isinstance(entries, list):
            sampled = random.sample(entries, k=min(20, len(entries)))
            text_block = "\n".join(["- " + s for s in sampled])
        else:
            text_block = entries
        return text_block

    def _perturbation_to_text(self, gene_list):
        """组合多个基因的文本表示"""
        genes = ' + '.join([gene for gene in gene_list])
        entries = '\n'.join([self._gene_to_text(gene) for gene in gene_list if gene != 'ctrl'])
        return prompt_perturbation.format(
            pert='SINGLE-GENE' if len(gene_list) == 1 else 'MUlTI-GENE',
            gene=genes,
            entries=entries,
            cell='None'
        )

    def tokenize(self, pert_texts):
         # tokenizer 编码
        tokenized = self.tokenizer(
            pert_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # forward 生成 embedding
        with torch.no_grad() if self.freeze_model else torch.enable_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)

            if self.model_name == 'BiomedBERT':
                hidden = outputs.hidden_states[-1]
                cls_embedding = hidden[:, 0]  
            else:
                last_hidden = outputs.last_hidden_state
                cls_embedding = last_hidden[:, 0]  
        return cls_embedding
    
        
    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        # 使用缓存数据，避免实时采样
        # 支持 str 或 list 的 batch_data.pert 表达形式
        if isinstance(batch_data.pert[0], str):
            batch_gene_lists = [[g for g in p.split('+') if g != 'ctrl'] for p in batch_data.pert]
        elif isinstance(batch_data.pert[0], list):
            batch_gene_lists = [[g for g in p if g != 'ctrl'] for p in batch_data.pert]
        else:
            raise ValueError("Unsupported format in batch_data.pert")
        gene_list =  [[gene] for gene_list in batch_gene_lists for gene in gene_list]
        index_list = [idx for i, gene_list in enumerate(batch_gene_lists) for idx in len(gene_list) * [i]]

        if self.fusion == "early":
            pert_texts = [self._perturbation_to_text(gene_list) for gene_list in batch_gene_lists]
            cls_embedding = self.tokenize(pert_texts)
        elif self.fusion == 'late':
            pert_texts = [self._perturbation_to_text(gene) for gene in gene_list]
            temp = self.tokenize(pert_texts)
            cls_embedding = torch.zeros(len(batch_data.pert), temp.shape[1], dtype=temp.dtype).to(temp.device)
            expanded_indices = torch.LongTensor(index_list).view(-1, 1).repeat(1, temp.shape[1]).to(temp.device)
            cls_embedding.scatter_add_(dim=0, index=expanded_indices, src=temp)
        else:
            pert_texts = [self._perturbation_to_text(gene_list) for gene_list in batch_gene_lists] + [self._perturbation_to_text(gene) for gene in gene_list]
            index_list = [i for i, _ in enumerate(batch_gene_lists)] + index_list

            temp = self.tokenize(pert_texts)
            cls_embedding = torch.zeros(len(batch_data.pert), temp.shape[1], dtype=temp.dtype).to(temp.device)
            expanded_indices = torch.LongTensor(index_list).view(-1, 1).repeat(1, temp.shape[1]).to(temp.device)
            cls_embedding.scatter_add_(dim=0, index=expanded_indices, src=temp)

        if not pred_flag:
            if noise_scale == None:
                noise_scale = self.config.noise_scale
            mask = (torch.rand_like(cls_embedding) > 0).float()
            cls_embedding = cls_embedding.float() + torch.rand_like(cls_embedding.float()) * noise_scale * mask
        else:
            cls_embedding = cls_embedding.float()

        return self.adaptor(cls_embedding.float())


class GeneLearner(nn.Module):
    def __init__(self, config, G_go, G_go_weight):

        super(GeneLearner, self).__init__()

        self.device = config.device
        self.num_perts = config.num_perts
        # this dict contains all tensors needed to be optimized
        self.num_layers = 3
        self.config = config
        # pert_emb
        self.pert_emb = nn.Embedding(config.num_perts, config.hidden_dim, max_norm=True)
        torch.nn.init.kaiming_normal_(self.pert_emb.weight)

        # pert encoder
        ### perturbation gene ontology GNN
        self.G_sim = G_go.to(self.device)
        self.G_sim_weight = G_go_weight.to(self.device)
        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers):
            self.sim_layers.append(SGConv(config.hidden_dim, config.hidden_dim, 1))
        for m in self.sim_layers.modules():
            if isinstance(m, SGConv):
                nn.init.xavier_uniform_(m.lin.weight)  # Xavier Uniform
                nn.init.zeros_(m.lin.bias)  # Bias 设置为零

        self.pert_fuse = MLP([config.hidden_dim, config.hidden_dim, config.aux_dim], last_layer_act='ReLU')
        for m in self.pert_fuse.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            
    def forward(self, batch_data, pred_flag=True, noise_scale=None):
        pert_idx = batch_data.pert_idx
        pert_index = []
        for i, a in enumerate(pert_idx):
            for b in a:
                if b != -1:
                    pert_index.append([i, b])
        pert_index = torch.tensor(pert_index).T
                    
        # 3. perturbation embedding
        pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.device))

        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, ) # self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        if pert_index.shape[0] != 0:
            pert_track = {}
            for a, b in enumerate(pert_index[0]):
                if b.item() in pert_track:
                    pert_track[b.item()] = pert_track[b.item()] + pert_global_emb[pert_index[1][a]]
                else:
                    pert_track[b.item()] = pert_global_emb[pert_index[1][a]]

            if len(list(pert_track.values())) > 0:
                if len(list(pert_track.values())) == 1:
                    # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                else:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

        return emb_total


def task_learner_selection(config, task, emb_flag, device, G_disc=None, G_go=None, G_go_weight=None, adata=None):
    if task == 'gene_knockout':
        if emb_flag == 'qwen':
            task_learner = FreezedEmbeddingLearner(config, device, model_name='qwen')
        if emb_flag == 'ESM2':
            task_learner = FreezedEmbeddingLearner(config, device, model_name='ESM2')
        if emb_flag == 'GENEPT':
            task_learner = FreezedEmbeddingLearner(config,device, model_name='GENEPT')
        if emb_flag == 'emb':
            task_learner = EmbeddingLearner(config, G_disc, device)
        elif emb_flag == 'onehot':
            task_learner = OnehotLearner(config)
        elif emb_flag == 'gene':
            config.device = device
            task_learner = GeneLearner(config, G_go, G_go_weight)
    elif task == 'drug_perturbation':
        if emb_flag == 'Biomed':
            task_learner = ChemLearner(config, model_name='BiomedBERT')
        elif emb_flag == 'ChemBERT':
            task_learner = ChemLearner(config, model_name='ChemBERT')
        elif emb_flag == 'sep':
            task_learner = PromptTokenLearner(config, fusion='sep')
        elif emb_flag == "smile":
            task_learner = SmileLearner(config)
        else:
            task_learner = OnehotLearner(config)
    else:
        raise ValueError('task should be gene_knockout or drug_perturbation')
    
    return task_learner