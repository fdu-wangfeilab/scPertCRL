import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SGConv
from torch.distributions import Normal
from torch.autograd import Variable
from torch_geometric.utils import to_dense_batch

from DAGPert.model.model_utils import *
from DAGPert.model.model_base import *
from DAGPert.model.model_ge import *



class VAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # for parameter record
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.aux_dim = config.aux_dim
        self.latent_dim = config.latent_dim

        self.encoder = VarFclEncoder(
                        input_dim = self.input_dim,
                        hidden_dim = self.hidden_dim,
                        enc_dim = self.latent_dim,
                        depth = 4, 
                        )
            
        self.prior = VarFclEncoder(
                input_dim = self.aux_dim,
                hidden_dim = self.hidden_dim,
                enc_dim = self.latent_dim,
                depth = 4, 
                )

        self.decoder = FclDecoder(
                            enc_dim = self.latent_dim, 
                            hidden_dim = self.hidden_dim,
                            output_dim = self.input_dim,
                            depth = 4,
                            )
        
    
    def forward(self, x, u):
        mu, var = self.encoder(x) 
        z = self.encoder.reparameterize(mu, var)
        recon = self.decoder(z)

        z_u = z
        pred = self.decoder(z_u)

        return recon, pred, None, z, (mu, var)

class addVAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # for parameter record
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.aux_dim = config.aux_dim
        self.latent_dim = config.latent_dim

        self.encoder = VarFclEncoder(
                        input_dim = self.input_dim,
                        hidden_dim = self.hidden_dim,
                        enc_dim = self.latent_dim,
                        depth = 4, 
                        )
            
        self.prior = VarFclEncoder(
                input_dim = self.aux_dim,
                hidden_dim = self.hidden_dim,
                enc_dim = self.latent_dim,
                depth = 4, 
                )

        self.decoder = FclDecoder(
                            enc_dim = self.latent_dim, 
                            hidden_dim = self.hidden_dim,
                            output_dim = self.input_dim,
                            depth = 4,
                            )
        
    
    def forward(self, x, u):
        mu, var = self.encoder(x) 

        prior_mu, prior_var = self.prior(u)
        z = self.encoder.reparameterize(mu, var)
        recon = self.decoder(z)

        z_u = self.encoder.reparameterize(mu+prior_mu.unsqueeze(1), var+prior_var.unsqueeze(1))
        pred = self.decoder(z_u)

        return recon, pred, None, z, (mu, var)

class DynamicDAGVAE(nn.Module):
    def __init__(self, config,
                 base_net,
                 G_graph,
                 gene_name,
                 version='v2') -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.aux_dim = config.aux_dim
        self.latent_dim = config.latent_dim
        self.var_dim = config.var_dim
        self.ablation_setting = config.ablation_setting
        print("!!!ablation setting:", self.ablation_setting)

        self.encoder = VarFclEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim * 4,
            enc_dim=self.latent_dim,
            depth=config.depth * 2,
            var_dim=config.var_dim,
            dropout_rate=config.dropout_rate,
        )
        self.decoder = Gene_Specific_Decoder_ZIG(
            in_dims=self.latent_dim,
            hidden_size=self.hidden_dim // 4,
            var_dim=self.var_dim,
            nonlinear=False
        )

        self.iencoder = VarFclEncoder(
            input_dim=self.aux_dim,
            hidden_dim=self.hidden_dim,
            enc_dim=self.latent_dim,
            var_dim=self.var_dim,
            depth=1,
        )


        if version == 'v2':
            self.generator = DSCM_v2(
                domain_dim=self.aux_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                var_dim=self.var_dim,
                depth=config.depth,
                ref_base_net=base_net, 
                G_graph=G_graph,
                k = config.k_neumann,
                net_type=config.net_type,
                rank=config.rank,
                nonlinear_flag=config.nonlinear_flag,
            )
        else:
            self.generator = DSCM(
                domain_dim=self.aux_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                var_dim=self.var_dim,
                depth=config.depth,
                ref_base_net=base_net,
                G_graph=G_graph,
                k = config.k_neumann,
                net_type=config.net_type,
                rank=config.rank,
                nonlinear_flag=config.nonlinear_flag,
            )


        self.cls = TargetClassifier(
            input_dim=self.latent_dim + config.set_flag * self.var_dim, 
            hidden_dim=self.hidden_dim,
            output_dim=1, 
        )

        self.prior = VarFclEncoder(
            input_dim=self.aux_dim,
            hidden_dim=self.hidden_dim,
            enc_dim=self.latent_dim,
            var_dim=self.var_dim,
            depth=1,
        )

    def pre_forward(self, batch_data, u, training_flag=True):
        x, _ = to_dense_batch(batch_data.x, batch_data.batch)

        u_ext = u.unsqueeze(1).repeat(1, x.size(1), 1) 

        mu, var = self.encoder(x)
        p_mu, p_var = self.prior(u_ext)
        
        z = self.encoder.reparameterize(mu, var) if training_flag else mu
        recon, base_coff = self.generator(z)
        recon = self.decoder(recon)

        return recon, base_coff, z, (mu, var), (p_mu, p_var) 
        
    def forward(self, batch_data, u, training_flag=True):
        batch_size = len(batch_data.pert)
        x, _ = to_dense_batch(batch_data.x, batch_data.batch)


        u_ext = u.unsqueeze(1).repeat(1, x.size(1), 1) 

        mu, var = self.encoder(x)
        
        z = self.encoder.reparameterize(mu, var) if training_flag else mu
        recon, base_coff = self.generator(z)
        recon = self.decoder(recon)

        i_mu, i_var = self.iencoder(u_ext)


        z = (z + self.iencoder.reparameterize(i_mu, i_var)) if training_flag else (z + i_mu)

        pred_z, coff = self.generator(z, u)

        pred = self.decoder(pred_z)

        return recon, pred, (base_coff, coff), z, (mu, var), (i_mu, i_var) 

    def target_predict(self, task_emb=None, noise=None, weight=None):
        B, N, _ = weight.shape

        noise = noise.transpose(2, 1)
        noise = noise.reshape(B, N, -1)
        weight = weight.reshape(B, N, -1)

        h = torch.concat([noise, weight], dim=-1)
        out = self.cls(h)
        return F.sigmoid(out.squeeze(-1))



