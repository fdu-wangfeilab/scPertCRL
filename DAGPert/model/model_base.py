import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import SGConv
from torch.distributions import Normal
from torch.autograd import Variable
from DAGPert.model.model_utils import *
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM


class VarFclEncoder(nn.Module):
    """
    Encoder using fully connected layer 
    """
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 enc_dim,
                 dropout_rate: float=0.1, 
                 var_dim=None,
                 depth: int=2) -> None:
        """
        Parameters
        ----------
        input_dim
            input dimension
        hidden_dim

        """
        super().__init__()
        
        self.enc_dim = enc_dim
        self.var_dim = var_dim
        self.mlp = NetBlock(
            nlayer=depth,
            dim_list=[input_dim] + [hidden_dim] * depth,
            act_list=[nn.GELU()] * depth,
            dropout_rate=dropout_rate,
            norm_flag = 'ln',
        )

        if self.var_dim == None:
            self.var_enc = nn.Sequential(
            nn.Linear(hidden_dim, enc_dim)
            ) 
            self.mu_enc = nn.Sequential(
                nn.Linear(hidden_dim, enc_dim)
            ) 
        else:
            self.var_enc = nn.Sequential(
                nn.Linear(hidden_dim, enc_dim * var_dim)
            ) 
            self.mu_enc = nn.Sequential(
                nn.Linear(hidden_dim, enc_dim * var_dim)
            ) 

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()
   
    def forward(self, x):
        h = self.mlp(x)
        mu = self.mu_enc(h)
        var = torch.clamp(torch.exp(self.var_enc(h)), min=1e-20) 
        if self.var_dim is None:
            return mu, var
        else:
            batch_shape = mu.shape[:-1]
            return mu.reshape(*batch_shape, self.enc_dim, self.var_dim), \
                    var.reshape(*batch_shape, self.enc_dim, self.var_dim)


class FclDecoder(nn.Module):
    def __init__(self, enc_dim: int, hidden_dim: int, output_dim: int,
                 depth: int):

        super(FclDecoder, self).__init__()

        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth

        self.input_layer = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.blks = nn.ModuleList()
        for _ in range(depth):
            self.blks.append(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ))
            
        self.output_layer = nn.Sequential(
           nn.Linear(hidden_dim, output_dim),
        ) 

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) == 4:
            inputs = inputs.squeeze(-1) 
        h = self.input_layer(inputs)
        for blk in self.blks:
            h = blk(h)
        
        preds = self.output_layer(h)
        return preds


class DSCM(nn.Module):
    def __init__(self, latent_dim: int, domain_dim: int,
                 hidden_dim: int, var_dim:int, depth: int, rank=48, nonlinear_flag=True, ref_base_net=None, G_graph=None, k=5, net_type='cond'):

        super(DSCM, self).__init__()
        self.base_net = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001 * (torch.ones(latent_dim, latent_dim) - torch.eye(latent_dim)))
        if ref_base_net is not None and latent_dim == ref_base_net.shape[0]:
            print('initialise base net with reference net')
            prior_mask_tensor = torch.tensor(ref_base_net, dtype=torch.float32)
            self.base_net = nn.Parameter(prior_mask_tensor * 0.1) 
        
        self.G_graph = G_graph

        if net_type == 'cond':
            self.coeff_net = CondLoraLinear(domain_dim, latent_dim, rank=rank)
        else:
            self.coeff_net = CondLoraLinear_v2(domain_dim, latent_dim, rank=rank)
        # self.path_expander = PathExpander(latent_dim=latent_dim, var_dim=var_dim, max_steps=k, hidden_dim=hidden_dim, nonlinear_flag=nonlinear_flag)

        self.ref_base_net = ref_base_net
        self.domain_dim = domain_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depth = depth


    def adj_inv(self, coeffs):
        mask = Variable(torch.from_numpy(np.ones(self.latent_dim) - \
                        np.eye(self.latent_dim)).float(), requires_grad=False).to(coeffs)
        I = torch.eye(self.latent_dim)

        batch_I = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        batch_mask = mask.repeat(coeffs.size(0), 1, 1).to(coeffs)
        
        I_coeffs = batch_I - coeffs * batch_mask # 构建I-W的矩阵

        eye_batch = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        coeffs_inv = torch.linalg.solve(I_coeffs, eye_batch)
        return coeffs_inv
    
    def adj_mask(self, coeffs):
        mask = Variable(torch.from_numpy(np.ones(self.latent_dim) - \
                        np.eye(self.latent_dim)).float(), requires_grad=False).to(coeffs)
        I = torch.eye(self.latent_dim)

        batch_I = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        batch_mask = mask.repeat(coeffs.size(0), 1, 1).to(coeffs)
        coeffs = coeffs * batch_mask
        return coeffs
    
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs, labels=None, pert_info=None):
        if labels is None:
            coeffs = self.base_net
            coeffs = self.base_net.unsqueeze(0).expand(inputs.shape[0], -1, -1)
            skeleton = None
        else:
            if self.G_graph is not None and pert_info is not None and self.latent_dim == self.ref_base_net.shape[0]:
                # print(pert_info)
                graph_matrices = [self.G_graph.get(pert, None) for pert in pert_info]
                dense_matrices = [
                    torch.Tensor(mat.toarray()) if hasattr(mat, "toarray") else torch.Tensor(mat)
                    for mat in graph_matrices
                ]
                skeleton = torch.stack(dense_matrices, dim=0).to(inputs.device)
                
                coeffs = self.base_net + self.coeff_net(labels).reshape(labels.shape[0], self.latent_dim, self.latent_dim) * skeleton 
            else:
                coeffs = self.base_net + self.coeff_net(labels).reshape(labels.shape[0], self.latent_dim, self.latent_dim)
            coeffs = torch.reshape(coeffs, (labels.shape[0], self.latent_dim, self.latent_dim))
        
        I = torch.eye(self.latent_dim, device = inputs.device)
        mask = torch.ones(self.latent_dim, self.latent_dim, device = inputs.device) - I
        W = coeffs * mask

        # print(inputs.shape)
        preds = torch.linalg.solve(
            (I - W).transpose(1, 2),   # (B, d, d)
            inputs.squeeze(-1).transpose(1,2)     # (B, c, d, 1)
        ).transpose(1,2)

        return preds, coeffs
    
    def h_func(self, coeffs):
        if len(coeffs.shape) == 3:
            W = torch.mean(coeffs, 0)
        else:
            W = coeffs
        E = torch.matrix_exp(torch.clamp(W * W, max=10))
        h = torch.trace(E) - self.latent_dim
        return h
    
    def w_l1_reg(self, coeffs):
        reg = torch.mean(torch.abs(coeffs))
        return reg
    
    def l2_reg(self, coeffs):
        reg = 0.
        if len(coeffs.shape) == 3:
            weight = torch.mean(coeffs, 0)
        else:
            weight = coeffs
        reg += torch.sum(weight ** 2)
        return reg


class DSCM_v2(nn.Module):
    def __init__(self, latent_dim: int, domain_dim: int,
                 hidden_dim: int, var_dim:int, depth: int, rank=48, nonlinear_flag=True, ref_base_net=None, G_graph=None, k=5, net_type='cond'):

        super(DSCM_v2, self).__init__()
        self.base_net = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001 * (torch.ones(latent_dim, latent_dim) - torch.eye(latent_dim)))
        if ref_base_net is not None and latent_dim == ref_base_net.shape[0]:
            print('initialise base net with reference net')
            prior_mask_tensor = torch.tensor(ref_base_net, dtype=torch.float32)
            self.base_net = nn.Parameter(prior_mask_tensor * 0.1) 
        
        self.G_graph = G_graph

        # attention-based graph learner
        if net_type == 'cond':
            self.coeff_net = CondLoraLinear(domain_dim, latent_dim, rank=rank)
        else:
            self.coeff_net = CondLoraLinear_v2(domain_dim, latent_dim, rank=rank)

        self.path_expander = PathExpander(latent_dim=latent_dim, var_dim=var_dim, max_steps=k, hidden_dim=hidden_dim, nonlinear_flag=nonlinear_flag)

        self.ref_base_net = ref_base_net
        self.domain_dim = domain_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.depth = depth


    def adj_inv(self, coeffs):
        mask = Variable(torch.from_numpy(np.ones(self.latent_dim) - \
                        np.eye(self.latent_dim)).float(), requires_grad=False).to(coeffs)
        I = torch.eye(self.latent_dim)

        batch_I = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        batch_mask = mask.repeat(coeffs.size(0), 1, 1).to(coeffs)
        
        I_coeffs = batch_I - coeffs * batch_mask # 构建I-W的矩阵

        eye_batch = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        coeffs_inv = torch.linalg.solve(I_coeffs, eye_batch)
        return coeffs_inv
    
    def adj_mask(self, coeffs):
        mask = Variable(torch.from_numpy(np.ones(self.latent_dim) - \
                        np.eye(self.latent_dim)).float(), requires_grad=False).to(coeffs)
        I = torch.eye(self.latent_dim)

        batch_I = I.repeat(coeffs.size(0), 1, 1).to(coeffs)
        batch_mask = mask.repeat(coeffs.size(0), 1, 1).to(coeffs)
        coeffs = coeffs * batch_mask
        return coeffs
    
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs, labels=None, pert_info=None):
        if labels is None:
            coeffs = self.base_net
            coeffs = self.base_net.unsqueeze(0).expand(inputs.shape[0], -1, -1)
            skeleton = None
        else:
            if self.G_graph is not None and pert_info is not None and self.latent_dim == self.ref_base_net.shape[0]:
                # print(pert_info)
                graph_matrices = [self.G_graph.get(pert, None) for pert in pert_info]
                dense_matrices = [
                    torch.Tensor(mat.toarray()) if hasattr(mat, "toarray") else torch.Tensor(mat)
                    for mat in graph_matrices
                ]
                skeleton = torch.stack(dense_matrices, dim=0).to(inputs.device)
                
                coeffs = self.base_net + self.coeff_net(labels).reshape(labels.shape[0], self.latent_dim, self.latent_dim) * skeleton 
            else:
                coeffs = self.base_net + self.coeff_net(labels).reshape(labels.shape[0], self.latent_dim, self.latent_dim)
            coeffs = torch.reshape(coeffs, (labels.shape[0], self.latent_dim, self.latent_dim))
        
        # coeffs_inv = self.adj_inv(coeffs)
        # preds = torch.bmm(inputs.unsqueeze(1), coeffs_inv).squeeze(1)
        preds, step_outputs = self.path_expander(inputs, coeffs)
        return preds, coeffs
    
    def h_func(self, coeffs):
        if len(coeffs.shape) == 3:
            W = torch.mean(coeffs, 0)
        else:
            W = coeffs
        E = torch.matrix_exp(torch.clamp(W * W, max=10))
        h = torch.trace(E) - self.latent_dim
        return h
    
    def w_l1_reg(self, coeffs):
        reg = torch.mean(torch.abs(coeffs))
        return reg
    
    def l2_reg(self, coeffs):
        reg = 0.
        if len(coeffs.shape) == 3:
            weight = torch.mean(coeffs, 0)
        else:
            weight = coeffs
        reg += torch.sum(weight ** 2)
        return reg


class MLP(nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.LayerNorm(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class Gene_Specific_Decoder(nn.Module):
    def __init__(self, 
                 in_dims, 
                 hidden_size=16, 
                 var_dim=4, 
                 num_layers=4,
                 dropout_rate=0.2,
                 nonlinear=False):
        super(Gene_Specific_Decoder, self).__init__()
        
        self.num_genes = in_dims
        self.hidden_size = hidden_size 
        self.nonlinear = nonlinear
        self.var_dim = var_dim
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.randn(self.num_genes, var_dim, hidden_size))
        self.indv_b1 = nn.Parameter(torch.zeros(self.num_genes, hidden_size))
        nn.init.xavier_normal_(self.indv_w1)

        # 多层 MLP，每层加 LayerNorm
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),   # ✅ LayerNorm
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ))

        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.randn(self.num_genes, hidden_size, 1))
        self.indv_b2 = nn.Parameter(torch.zeros(self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1, x.size()[1], x.size()[2], 1)
        hidden = torch.einsum('bsgi,gih->bsgh', x, self.indv_w1)  
        hidden = hidden + self.indv_b1.unsqueeze(0).unsqueeze(0) 
        if self.nonlinear:
            hidden = torch.relu(hidden)
        
        # 多层 MLP + LayerNorm
        for layer in self.layers:
            hidden = layer(hidden)

        output = torch.einsum('bsgh,ghd->bsgd', hidden, self.indv_w2)  # [8, 50, 2443, 1]
        output = output.squeeze(-1) + self.indv_b2.unsqueeze(0).unsqueeze(0)  # [8, 50, 2443]
        if self.nonlinear:
            output = torch.relu(output)      

        return output


class Gene_Specific_Decoder_ZIG(nn.Module):
    def __init__(self, 
                 in_dims, 
                 hidden_size=16, 
                 var_dim=4, 
                 dropout_rate=0.2,
                 nonlinear=False):
        super(Gene_Specific_Decoder_ZIG, self).__init__()
        
        self.num_genes = in_dims
        self.hidden_size = hidden_size 
        self.nonlinear = nonlinear
        self.var_dim = var_dim

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.randn(self.num_genes, var_dim, hidden_size))
        self.indv_b1 = nn.Parameter(torch.zeros(self.num_genes, hidden_size))
        nn.init.xavier_normal_(self.indv_w1)

        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.randn(self.num_genes, hidden_size, 1))
        self.indv_b2 = nn.Parameter(torch.zeros(self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)

        # final gene specific decoder
        self.indv_w3 = nn.Parameter(torch.randn(self.num_genes, hidden_size, 1))
        self.indv_b3 = nn.Parameter(torch.zeros(self.num_genes))
        nn.init.xavier_normal_(self.indv_w3)

        # final gene specific decoder
        self.indv_w4 = nn.Parameter(torch.randn(self.num_genes, hidden_size, 1))
        self.indv_b4 = nn.Parameter(torch.zeros(self.num_genes))
        nn.init.xavier_normal_(self.indv_w4)


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(-1, x.size()[1], x.size()[2], 1)
        hidden = torch.einsum('bsgi,gih->bsgh', x, self.indv_w1)  
        hidden = hidden + self.indv_b1.unsqueeze(0).unsqueeze(0) 
        
        hidden = F.dropout(hidden, p=0.1, training=self.training)
        hidden = F.relu(hidden) 

        output = torch.einsum('bsgh,ghd->bsgd', hidden, self.indv_w2)  # [8, 50, 2443, 1]
        output = output.squeeze(-1) + self.indv_b2.unsqueeze(0).unsqueeze(0)  # [8, 50, 2443]
        output = F.relu(output) 
        
        # --- logσ head ---
        log_sigma = torch.einsum('bsgh,ghd->bsgd', hidden, self.indv_w3).squeeze(-1)
        log_sigma = log_sigma + self.indv_b3.unsqueeze(0).unsqueeze(0)
        log_sigma = torch.clamp(log_sigma, min=-5, max=5)  
   

        # --- logitπ head ---
        logit_pi = torch.einsum('bsgh,ghd->bsgd', hidden, self.indv_w4).squeeze(-1)
        logit_pi = logit_pi + self.indv_b4.unsqueeze(0).unsqueeze(0)
        logit_pi = logit_pi  # 初始更偏稀疏
        
        return output, log_sigma, logit_pi


class NetBlock(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list, 
        dropout_rate: float, 
        noise_rate: float=0,
        norm_flag: str = 'ln',
        ):
        """
        multiple layers netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
        
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.norm_flag = norm_flag
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            if norm_flag == 'ln':
                self.bn_list.append(nn.LayerNorm(dim_list[i + 1]))
            elif norm_flag == 'bn':
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))

        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            if self.norm_flag != None:
                x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)
        return x


class PathExpander(nn.Module):
    def __init__(self, latent_dim, var_dim=1, max_steps=3, hidden_dim=128, residual_tol=1e-3, nonlinear_flag=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.residual_tol = residual_tol
        self.var_dim = var_dim
        self.nonlinear_flag = nonlinear_flag

        # 每阶一个非线性映射（可以视为每阶路径的激活方式）
        self.phi_k = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.var_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, self.var_dim),
                # nn.LeakyReLU(),
            ) for _ in range(max_steps + 1)
        ])


    
    def forward(self, inputs, coeffs, graph_mask=None):
        """
        inputs: (B, V) or (B, V, D) - 输入数据
        其中 V 是基因数，D 是特征维度
        coeffs: (B, V, V) - W matrix
        graph_mask: (B, V, V) or (V, V) - optional binary mask
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)  # (B, V, 1)
        B, S, V, D = inputs.shape
    
        device = inputs.device

        if graph_mask is not None:
            if graph_mask.dim() == 2:
                graph_mask = graph_mask.unsqueeze(0).expand(B, -1, -1)
            coeffs = coeffs * graph_mask.to(coeffs)

        X = inputs  # (B, V, D)
        step_outputs = []
        X_prev = None

        for k in range(self.max_steps + 1):
            if k == 0:
                X_k = X.clone()  # I项
            else:
                X_k = torch.einsum('bsvd,bwv->bswd', X_k, coeffs)
 
            if self.nonlinear_flag:
                trans_Xk = self.phi_k[k](X_k)
            else:
                trans_Xk = X_k
            step_outputs.append(trans_Xk)

            if k > 0:
                residual = torch.norm(X_k - X_prev, dim=(1, 2))  # 计算线性传播残差
                if (residual < self.residual_tol).all():
                    break
            X_prev = X_k 
            
        # 最终输出为各阶非线性变换结果之和
        full_output = torch.stack(step_outputs, dim=0).sum(dim=0)
        return full_output, step_outputs


class TargetClassifier(nn.Module):
    """
    Encoder using fully connected layer 
    """
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim=1,
                 dropout_rate: float=0.8) -> None:
        """
        Parameters
        ----------
        input_dim
            input dimension
        hidden_dim

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(0.8)
        self.mlp = NetBlock(
            nlayer=1,
            dim_list=[input_dim, 128],
            act_list=[nn.GELU()],
            dropout_rate=dropout_rate,
            norm_flag = None,
        )
        self.cls = nn.Linear(128, output_dim)

    def forward(self, x):
        h = self.dropout(x)
        h = self.mlp(h)
        out = self.cls(h)

        return out
    

class CondLoraLinear(nn.Module):
    def __init__(self, domain_dim, latent_dim, rank=48, scale=0.001): # 0.001
        super().__init__()
        self.domain_dim = domain_dim
        self.latent_dim = latent_dim
        self.rank = rank
        self.scale = scale
        self.A_net = nn.Linear(domain_dim, latent_dim * rank)
        self.B_net = nn.Linear(domain_dim, latent_dim * rank)


    def forward(self, u):
        A = self.A_net(u)
        B = self.B_net(u)

        A = A.view(-1, self.latent_dim, self.rank) #[B, N, 1]
        B = B.view(-1, self.latent_dim, self.rank) #[B, N, 1]

        delta_w = torch.matmul(A, B.transpose(-1, -2))
        delta_w *= self.scale
        return delta_w


class CondLoraLinear_v2(nn.Module):
    def __init__(self, domain_dim, latent_dim, rank=48, scale=0.001): # 48
        super().__init__()
        self.A_net = nn.Sequential(
                                nn.Linear(domain_dim, domain_dim // 4),
                                nn.ReLU(),
                                nn.Linear(domain_dim // 4, latent_dim * rank)
                            )
        self.B_net = nn.Sequential(
                                nn.Linear(domain_dim, domain_dim // 4),
                                nn.ReLU(),
                                nn.Linear(domain_dim // 4, latent_dim * rank)
                            )
        self.rank = rank
        self.latent_dim = latent_dim
        self.scale = scale

    def forward(self, u):
        A = self.A_net(u)
        B = self.B_net(u)

        A = A.view(-1, self.latent_dim, self.rank) #[B, N, 1]
        B = B.view(-1, self.latent_dim, self.rank) #[B, N, 1]

        delta_w = torch.matmul(A, B.transpose(-1, -2))
        delta_w *= self.scale
        return delta_w


