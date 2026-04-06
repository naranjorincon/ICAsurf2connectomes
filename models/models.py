import sys
sys.path.append('../') #models
sys.path.append('./') # utils folders
sys.path.append('../../')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
# from vit_pytorch.vit import Transformer #See here: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# from utils import utils
# from models.models_helper_fcn import * # pretty sure this is for DDPM
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# helper classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5 #square root of d_k and it is used as denominator hence **-0.5
        self.norm = nn.LayerNorm(dim) #layer norm instead of batch norm
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 heads, 
                 dim_head, 
                 mlp_dim, 
                 dropout = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class EncoderSiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        output_length = 512,
                        num_channels = 15,
                        num_patches = 320,
                        num_vertices = 153,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        ):

        super().__init__()
        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim
        dim_head = int(math.ceil(dim / heads))

        self.output_length = output_length
        self.dim = dim

        # inputs has size = b * c * n * v where b = batch, c = channels, f = features, n=patches, v=verteces
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # See here: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

        self.linear = nn.Linear(num_patches * dim, output_length * dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :] # was originally sliced by [:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Reshape the input tensor to (batch_size, num_patches * dim)
        latent = x.view(b, -1)
        # Apply the linear layer
        output = self.linear(latent)
        # Reshape the output tensor to (batch_size, output_length, dim)
        output = output.view(b, self.output_length, self.dim)

        return output, latent
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 d_model, 
                 nhead, 
                 dim_feedforward, 
                 dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.d_model = d_model
        
        self.masked_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked Multi-Head Attention
        tgt2, masked_attn_weights = self.masked_multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)  # Residual connection
        tgt = self.norm1(tgt)
        
        # Cross-Multi-Head Attention
        tgt2, cross_attn_weights = self.cross_multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)  # Residual connection
        tgt = self.norm2(tgt)
        
        # Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt2)  # Residual connection
        tgt = self.norm3(tgt)
        
        return tgt, masked_attn_weights, cross_attn_weights
    
class SurfaceImageTransformer(nn.Module):
    def __init__(self, *,
                        dim=384, 
                        depth=6,
                        heads=4,
                        num_patches=320,
                        upper_tri=4950, #parcellation
                        num_channels=15,
                        num_vertices=153,
                        dim_head=64,
                        dropout=0.1,
                        emb_dropout=0.3,
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_patches*dim),
            nn.GELU(),
            nn.Linear(num_patches*dim, upper_tri)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, img):
        out, latent = self.encoder(img) #out is BxCxPxV, latent is BxPxD=384, or what ever the altent dim is 384 is from SiT so arbitraty for BGT
        return out, latent
    
    def forward(self, img):
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        latent = self.to_latent(x) # collapses into a vector to extract mean and var

        final_x = self.mlp_head(latent)

        return final_x, latent
    
class SurfaceImageTransformer_experimentDecode(nn.Module):
    def __init__(self, *,
                        dim=192, 
                        depth=12,
                        heads=3,
                        num_patches=320,
                        upper_tri=4950, #parcellation
                        num_channels=15,
                        num_vertices=153,
                        dim_head=64,
                        dropout=0.1,
                        emb_dropout=0.3,
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_patches*dim),
            # nn.GELU(),
            # nn.Linear(num_patches*dim, num_patches*dim),
            # nn.GELU(),
            # nn.Linear(num_patches*dim, num_patches*dim),
            nn.GELU(),
            nn.Linear(num_patches*dim, upper_tri)
            # nn.Tanh() # just call it cause it applied element wise, but how does that change with diff dims? Find out at some point
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, img):
        out, latent = self.encoder(img) #out is BxCxPxV, latent is BxPxD=384, or what ever the altent dim is 384 is from SiT so arbitraty for BGT
        return out, latent
    
    def forward(self, img):
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        latent = self.to_latent(x) # collapses into a vector to extract mean and var

        final_x = self.mlp_head(latent)

        return final_x, latent
    
class SurfaceImageTransformer_VAE(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        num_patches = 320,
                        upper_tri=4950, #parcellation
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.,
                        VAE_latent_dim=10e3,
                        latent_samples=1000
                        ):

        super().__init__()
        self.latent_samples = latent_samples
        self.VAE_latent_dim = VAE_latent_dim
        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim #4 times embedding dimension according to DeT (data efficient transformers)
        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #+1 to num_patches for original cause regression token
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))

        self.fc_mu = nn.Linear(num_patches*dim, self.VAE_latent_dim) # linear project from batch x 10k -> batch x vae_latent
        self.fc_var = nn.Linear(num_patches*dim, self.VAE_latent_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.VAE_latent_dim),
            nn.GELU(),
            nn.Linear(self.VAE_latent_dim, upper_tri)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        b,c,n,v = img.shape
        x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        x += self.pos_embedding#[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_latent(x) # collapses into a vector to extract mean and var

        # reparam trick
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var) # make into std

        epsilon = torch.randn(self.latent_samples, b, self.VAE_latent_dim) #torch.randn_like(mu) # think its supposed to be mu
        z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
        z_average = z_samples.mean(dim=0)
        latent_step_identity = z_average
        # last step, FFN
        final_x = self.mlp_head(latent_step_identity)
        return final_x, mu, log_var
    
class linear_enc_PCA(nn.Module):
    def __init__(self, *, 
                        collapsed_ICA_dim=15*320*153, # default is flattened ICAd15 ico02
                        parcellation_sz=100, #parcellation
                        emb_dropout = 0.5,
                        latent_dim=1000 #128 in kraken for shared latent space
                        ):

        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        # encode
        self.linear_enc_to_latent = nn.Linear(collapsed_ICA_dim, latent_dim)
        #decode
        upper_tri_from_parcellation = int(0.5 * (parcellation_sz * (parcellation_sz-1)))
        self.linear_dec_from_latent = nn.Linear(latent_dim, upper_tri_from_parcellation)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, img):
        z = self.linear_enc_to_latent(img)
        z = self.dropout(z)
        # last step
        upper_tri_pred = self.linear_dec_from_latent(z)
        return upper_tri_pred, z


class SurfaceImageTransformer_ICArecon(nn.Module):
    def __init__(self, *,
                        dim=384, 
                        depth=6,
                        heads=4,
                        num_patches=320,
                        upper_tri=4950, #parcellation
                        num_channels=15,
                        num_vertices=153,
                        dim_head=64,
                        dropout=0.1,
                        emb_dropout=0.3,
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)')) # to have a latent vector

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_patches*dim),
            nn.Linear(num_patches*dim, num_channels*num_patches*num_vertices)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, img):
        out, latent = self.encoder(img) #out is BxCxPxV, latent is BxPxD=384, or what ever the altent dim is 384 is from SiT so arbitraty for BGT
        return out, latent
    
    def forward(self, img):
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        latent = self.to_latent(x) # collapses into a vector to extract mean and var

        final_x = self.mlp_head(latent)

        return final_x, latent

class SurfaceImageTransformer_ICArecon_linear(nn.Module):
    def __init__(self, *,
                        dim=384, 
                        depth=6,
                        heads=4,
                        num_patches=320,
                        upper_tri=4950, #parcellation
                        num_channels=15,
                        num_vertices=153,
                        dim_head=64,
                        dropout=0.1,
                        emb_dropout=0.3,
                        ):

        super().__init__()
        collapsed_ICA_dim = num_channels*num_patches*num_vertices
        
        self.flatten_then_linear = nn.Sequential(
            Rearrange('b c n v  -> b (c n v)'),
            # nn.Linear(collapsed_ICA_dim, collapsed_ICA_dim),
            nn.Linear(collapsed_ICA_dim, num_patches*dim),
            )

        self.dropout = nn.Dropout(emb_dropout)

        self.last_linear = nn.Sequential(
            # nn.LayerNorm(num_patches*dim),
            # nn.Linear(collapsed_ICA_dim, collapsed_ICA_dim),
            nn.LayerNorm(num_patches*dim),
            nn.Linear(num_patches*dim, collapsed_ICA_dim),
            )
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # def encode(self, img):
    #     out, latent = self.encoder(img) #out is BxCxPxV, latent is BxPxD=384, or what ever the altent dim is 384 is from SiT so arbitraty for BGT
    #     return out, latent

    def forward(self, img):
        x = self.flatten_then_linear(img)
        z = self.dropout(x)
        final_x = self.last_linear(z)
        return final_x, z
    
class linear_translator(nn.Module):
    def __init__(self, *,
                        dim=384, 
                        depth=6,
                        heads=4,
                        num_patches=320,
                        upper_tri=4950, #parcellation
                        num_channels=15,
                        num_vertices=153,
                        dim_head=64,
                        dropout=0.1,
                        emb_dropout=0.3,
                        ):

        super().__init__()
        collapsed_ICA_dim = num_channels*num_patches*num_vertices
        
        self.flatten_then_linear = nn.Sequential(
            Rearrange('b c n v  -> b (c n v)'),
            nn.Linear(collapsed_ICA_dim, num_patches*dim),
            )

        self.dropout = nn.Dropout(emb_dropout)

        self.last_linear = nn.Sequential(
            nn.LayerNorm(num_patches*dim),
            nn.GELU(),
            nn.Linear(num_patches*dim, collapsed_ICA_dim),
            )
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
    def forward(self, img):
        x = self.flatten_then_linear(img)
        z = self.dropout(x)
        final_x = self.last_linear(z)
        return final_x, z
    
    
class SurfaceImageTransformer_VAE_ICArecon(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        heads,
                        num_patches = 320,
                        upper_tri=4950, #parcellation
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.,
                        VAE_latent_dim=10e3
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim #4 times embedding dimension according to DeT (data efficient transformers)
        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #+1 to num_patches for original cause regression token
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))

        self.fc_mu = nn.Linear(num_patches*dim, VAE_latent_dim) # linear project from batch x 10k -> batch x vae_latent
        self.fc_var = nn.Linear(num_patches*dim, VAE_latent_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(VAE_latent_dim),
            nn.Linear(VAE_latent_dim, num_channels*num_patches*num_vertices)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        x += self.pos_embedding#[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_latent(x) # collapses into a vector to extract mean and var

        # reparam trick
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var) # make into std
        epsilon = torch.randn_like(mu) # think its supposed to be mu
        latent_step_identity = mu + (std * epsilon) # reparam trick

        # last step, FFN
        final_x = self.mlp_head(latent_step_identity)
        return final_x, mu, log_var

# class SGT_VAE(nn.Module):
#     def __init__(self, *,
#                         dim, 
#                         depth,
#                         heads,
#                         num_patches = 320,
#                         upper_tri=4950, #parcellation
#                         num_channels =4,
#                         num_vertices = 2145,
#                         dim_head = 64,
#                         dropout = 0.,
#                         emb_dropout = 0.,
#                         VAE_latent_dim=10e3,
#                         latent_samples=1000 #Sampling
#                         ):

#         super().__init__()

#         self.latent_samples = latent_samples
#         self.VAE_latent_dim = VAE_latent_dim
#         patch_dim = num_channels * num_vertices
#         mlp_dim = 4*dim #4 times embedding dimension according to DeT (data efficient transformers)
#         # inputs has size = b * c * n * v
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c n v  -> b n (v c)'),
#             nn.Linear(patch_dim, dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #+1 to num_patches for original cause regression token
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))

#         self.fc_mu = nn.Linear(num_patches*dim, self.VAE_latent_dim) # linear project from batch x 10k -> batch x vae_latent
#         self.fc_var = nn.Linear(num_patches*dim, self.VAE_latent_dim)

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(self.VAE_latent_dim),
#             nn.GELU(),
#             nn.Linear(self.VAE_latent_dim, upper_tri)
#         )

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, img):
#         b,c,n,v = img.shape
#         x = self.to_patch_embedding(img)
#         # b, n, _ = x.shape

#         x += self.pos_embedding#[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)
#         x = self.to_latent(x) # collapses into a vector to extract mean and var

#         # reparam trick
#         mu = self.fc_mu(x)
#         log_var = self.fc_var(x)
#         std = torch.exp(0.5 * log_var) # make into std

#         epsilon = torch.randn(self.latent_samples, b, self.VAE_latent_dim) #torch.randn_like(mu) # think its supposed to be mu
#         z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
#         z_average = z_samples.mean(dim=0)
#         latent_step_identity = z_average
#         # last step, FFN
#         final_x = self.mlp_head(latent_step_identity)
#         return final_x, mu, log_var