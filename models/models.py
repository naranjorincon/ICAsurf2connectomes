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

# class SiT_BGT_VAE(nn.Module):
#     def __init__(self, 
#                  dim_model, 
#                  encoder_depth, 
#                  nhead, 
#                  decoder_input_dim, 
#                  decoder_depth, 
#                  VAE_latent_dim=10000, 
#                  latent_length=102, 
#                  num_channels=15, 
#                  num_patches=320, 
#                  num_verteces=153,
#                  dropout=0.1):
#         super(SiT_BGT_VAE, self).__init__()

#         self.dim_model = dim_model
#         self.input_dim = decoder_input_dim
#         self.latent_length = latent_length

#         self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
#         # self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

#         self.encoder = EncoderSiT(dim=dim_model, 
#                                   depth=encoder_depth, 
#                                   heads=nhead, 
#                                   num_channels=num_channels,  
#                                   num_patches=num_patches, 
#                                   num_vertices=num_verteces, 
#                                   dropout=dropout,
#                                   output_length=latent_length,
#                                   emb_dropout=0.1)
        
#         self.fc_mu = nn.Linear(dim_model * latent_length, VAE_latent_dim)
#         self.fc_var = nn.Linear(dim_model * latent_length, VAE_latent_dim)

#         self.vae_latent_to_encoder_out = nn.Linear(VAE_latent_dim, dim_model * latent_length)
        
#         decoder_dim_feedforward = 4 * decoder_input_dim
#         self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

#         self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=decoder_input_dim, kernel_size=1, groups=latent_length)
#         # num_out_nodes = 100
#         # extra_start_tokens=1
#         # self.projection = MaskedLinear(in_features=latent_length*dim_model, out_features=int((num_out_nodes * (num_out_nodes-1)) / 2), mask=create_mask(num_out_nodes=num_out_nodes, latent_length=latent_length, num_extra_start_tokens=extra_start_tokens))

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)


#     def encode(self, src):
#         x, latent = self.encoder(src)
#         x = x.view(x.size()[0], -1) # reshape to [b x model_dim * latent_length]

#         mu = self.fc_mu(x)
#         log_var = self.fc_var(x)

#         return [x, mu, log_var]
    
#     def decode(self, tgt, encoder_out, tgt_mask):    
#         b, _ = tgt.size()
        
#         # Project to high-dimensional space
#         tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
#         tgt = tgt.view(b, -1, self.dim_model)
                
#         # Apply positional encoding
#         # tgt = self.positional_encoding(tgt)

#         # Reparameterization trick to sample from latent space
#         mu = encoder_out[0]
#         log_var = encoder_out[1]
#         std = torch.exp(0.5 * log_var)
#         epsilon = torch.randn_like(std)
#         z = mu + (std * epsilon)

#         vae_in_encoder_space = self.vae_latent_to_encoder_out(z)
#         vae_in_encoder_space = vae_in_encoder_space.view(b, self.latent_length, self.dim_model)

#         for layer in self.decoder_layers:
#             tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=vae_in_encoder_space, tgt_mask=tgt_mask)

#         tgt = tgt.view(b, -1)
#         tgt = self.projection(tgt.unsqueeze(-1))

#         return tgt #torch.tanh(tgt) 


#     def forward(self, src, tgt, tgt_mask, dropout=0.1):
#         b, _ = tgt.size()

#         # Project to high-dimensional space
#         tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
#         tgt = tgt.view(b, -1, self.dim_model)
        
#         # Apply positional encoding
#         # tgt = self.positional_encoding(tgt)

#         encoder_out = self.encode(src)
        
#         # Reparameterization trick to sample from latent space
#         mu = encoder_out[1]
#         log_var = encoder_out[2]
#         std = torch.exp(0.5 * log_var)
#         epsilon = torch.randn_like(std)
#         z = mu + (std * epsilon)

#         vae_in_encoder_space = self.vae_latent_to_encoder_out(z)
#         vae_in_encoder_space = vae_in_encoder_space.view(b, self.latent_length, self.dim_model)

#         for layer in self.decoder_layers:
#             tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=vae_in_encoder_space, tgt_mask=tgt_mask)
        
#         tgt = tgt.view(b, -1)
#         tgt = self.projection(tgt.unsqueeze(-1))
        
#         return tgt.squeeze(), mu, log_var 

# class SiT_BGT(nn.Module):
#     def __init__(self, 
#                  dim_model, 
#                  encoder_depth, 
#                  nhead, 
#                  decoder_input_dim, 
#                  decoder_depth, 
#                  latent_length=102, 
#                  num_channels=15, 
#                  num_patches=320, 
#                  num_verteces=153,
#                  dropout=0.1):
#         super(SiT_BGT, self).__init__()

#         self.dim_model = dim_model
#         self.input_dim = decoder_input_dim
#         self.latent_length = latent_length
#         self.device = "cpu"

#         self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
#         # self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

#         self.encoder = EncoderSiT(dim=dim_model, 
#                                   depth=encoder_depth, 
#                                   heads=nhead, 
#                                   num_channels=num_channels,  
#                                   num_patches=num_patches, 
#                                   num_vertices=num_verteces, 
#                                   dropout=dropout,
#                                   output_length=latent_length,
#                                   emb_dropout=0.1)
        
#         # self.fc_mu = nn.Linear(dim_model * latent_length, VAE_latent_dim)
#         # self.fc_var = nn.Linear(dim_model * latent_length, VAE_latent_dim)

#         # self.vae_latent_to_encoder_out = nn.Linear(VAE_latent_dim, dim_model * latent_length)
        
#         decoder_dim_feedforward = 4 * decoder_input_dim
#         self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

#         self.projection = nn.Conv1d(in_channels=latent_length*dim_model, out_channels=decoder_input_dim, kernel_size=1, groups=latent_length)

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)


#     def encode(self, src):
#         x, latent = self.encoder(src)
#         x = x.view(x.size()[0], -1) # reshape to [b x model_dim * latent_length]

#         return [x, latent]
    
#     def decode(self, tgt, encoder_out, tgt_mask):    
#         b, _ = tgt.size()
        
#         # Project to high-dimensional space
#         tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
#         tgt = tgt.view(b, -1, self.dim_model)

#         # Reparameterization trick to sample from latent space
#         x = encoder_out[0]
#         latent = encoder_out[1]

#         # vae_in_encoder_space = self.vae_latent_to_encoder_out(z)
#         encoder_space = x.view(b, self.latent_length, self.dim_model)

#         for layer in self.decoder_layers:
#             tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_space, tgt_mask=tgt_mask)

#         tgt = tgt.view(b, -1)
#         tgt = self.projection(tgt.unsqueeze(-1))

#         return tgt #,latent


#     def forward(self, src, tgt, tgt_mask, dropout=0.1):
#         b, _ = tgt.size()

#         # Project to high-dimensional space
#         tgt = self.flatten_to_high_dim(tgt.unsqueeze(-1))
#         tgt = tgt.view(b, -1, self.dim_model)
        
#         # encoder_space = self.encode(src)
#         # latent = encoder_space
#         # Reparameterization trick to sample from latent space
#         # mu = encoder_out[0]
#         # log_var = encoder_out[1]
#         # std = torch.exp(0.5 * log_var)
#         # epsilon = torch.randn_like(std)
#         # z = mu + (std * epsilon)

#         encoder_out = self.encode(src)
#         # Reparameterization trick to sample from latent space
#         x = encoder_out[0]
#         latent = encoder_out[1] # latent is still extracted cause not like VAE, but still needed for Kraken losses

#         # vae_in_encoder_space = self.vae_latent_to_encoder_out(z) # vae needs this to go from bottle neck to latent*dim_model but here, because not a VAE its already that size so skip
#         encoder_space = x.view(b, self.latent_length, self.dim_model)


#         for layer in self.decoder_layers:
#             tgt, masked_attn_weights, cross_attn_weights = layer(tgt=tgt, memory=encoder_space, tgt_mask=tgt_mask)
        
#         tgt = tgt.view(b, -1)
#         tgt = self.projection(tgt.unsqueeze(-1))
        
#         return tgt.squeeze(), latent

class SiT_BGT_VAE(nn.Module):
    def __init__(self, 
                 dim_model, 
                 encoder_depth, 
                 nhead, 
                 decoder_input_dim, 
                 decoder_depth, 
                 VAE_latent_dim=10000, 
                 latent_length=102, 
                 num_channels=15, 
                 num_patches=320, 
                 num_verteces=153,
                 dropout=0.1):
        super(SiT_BGT_VAE, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.flatten_to_high_dim = nn.Linear(50, self.dim_model)
        #self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
        # self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=num_verteces, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        self.fc_mu = nn.Linear(dim_model * latent_length, VAE_latent_dim)
        self.fc_var = nn.Linear(dim_model * latent_length, VAE_latent_dim)

        self.vae_latent_to_encoder_out = nn.Linear(VAE_latent_dim, dim_model * latent_length)
        
        decoder_dim_feedforward = 4 * decoder_input_dim
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection_block = nn.Sequential(
            nn.LayerNorm(self.dim_model),
            nn.Linear(self.dim_model, 50)  # Predict next 50 values at each step
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        x, latent = self.encoder(src)
        x = x.view(x.size()[0], -1) # reshape to [b x model_dim * latent_length]

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [x, mu, log_var]

    def autoregressive_decode(self, encoder_out, start_tokens, total_len=5000, block_size=50, teacher=None, teacher_forcing_ratio=1.0):
        b = start_tokens.shape[0]
        device = start_tokens.device
        assert total_len % block_size == 0, "total_len must be divisible by block_size"

        num_blocks = total_len // block_size
        decoder_input = start_tokens.unsqueeze(1) #decoder_input: stores the running sequence of previously generated blocks; shape starts as (B, 1, 50) and grows to (B, 2, 50), ..., (B, 100, 50)

        outputs = []

        # VAE sampling
        mu, log_var = encoder_out[1], encoder_out[2]
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        # print(f"ecncoder shape: {z.shape}")
        memory = self.vae_latent_to_encoder_out(z).view(b, self.latent_length, self.dim_model)

        for i in range(1, num_blocks):
            # USE BELOW IF: only using previous block to predict next
            # last_block = decoder_input[:, -1, :]  # (B, 50)
            # tgt = self.flatten_to_high_dim(last_block).unsqueeze(1)  # (B, 1, dim_model)

            # for layer in self.decoder_layers:
            #     tgt, _, _ = layer(tgt=tgt, memory=memory)

            #  USE BELOW IF: using ALL previous blocks to predict next
            flat_input = decoder_input.view(b, -1, 50)  # (B, i, 50) where i is the number of blocks generated
            tgt = self.flatten_to_high_dim(flat_input)  # (B, i, dim_model) **this is the projection from 50-dim to model dim

            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()  # (i, i) causal mask
            for layer in self.decoder_layers:
                tgt, _, _ = layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask) # (B, i, dim_model) output

            last_hidden = tgt[:, -1, :]  # (B, dim_model) -> utilizes the last-generated block (now attention aware) to generate next 50
            next_block = self.projection_block(last_hidden)  # (B, 50)
            outputs.append(next_block)

            # Probabilistic teacher forcing
            if teacher is not None and i < teacher.shape[1] and torch.rand(1).item() < teacher_forcing_ratio:
                next_input = teacher[:, i, :]  # (B, 50)
            else:
                next_input = next_block

            decoder_input = torch.cat([decoder_input, next_input.unsqueeze(1)], dim=1) # appending next block to decoder input

        return torch.cat(outputs, dim=1)  # (B, 4950)

    def forward(self, src, full_target=None, teacher_forcing_ratio=1.0):
        encoder_out = self.encode(src)

        # Split full_target into blocks of 50
        if full_target is not None:
            blocks = full_target.view(full_target.shape[0], -1, 50)  # (B, 100, 50)
            start_tokens = blocks[:, 0, :]  # first block is the 50 start tokens
            targets = blocks[:, 1:, :]  # the remaining 99 blocks to be predicted, so using first 50 to prefict all 4950
        else:
            start_tokens = torch.ones(src.size(0), 50, device=src.device)
            targets = None

        predictions = self.autoregressive_decode(
            encoder_out,
            start_tokens=start_tokens,
            total_len=5000,
            block_size=50,
            teacher=targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return predictions, encoder_out[1], encoder_out[2]
    
class SiT_BGT(nn.Module):
    def __init__(self, 
                 dim_model, 
                 encoder_depth, 
                 nhead, 
                 decoder_input_dim, 
                 decoder_depth, 
                #  VAE_latent_dim=10000, 
                 latent_length=102, 
                 num_channels=15, 
                 num_patches=320, 
                 num_verteces=153,
                 dropout=0.1):
        super(SiT_BGT, self).__init__()

        self.dim_model = dim_model
        self.input_dim = decoder_input_dim
        self.latent_length = latent_length

        self.flatten_to_high_dim = nn.Linear(50, self.dim_model)
        #self.flatten_to_high_dim = nn.Conv1d(in_channels=decoder_input_dim, out_channels=latent_length*dim_model, kernel_size=1, groups=latent_length)
        # self.positional_encoding = PositionalEncoding(d_model=dim_model, seq_len=latent_length, dropout=dropout)

        self.encoder = EncoderSiT(dim=dim_model, 
                                  depth=encoder_depth, 
                                  heads=nhead, 
                                  num_channels=num_channels,  
                                  num_patches=num_patches, 
                                  num_vertices=num_verteces, 
                                  dropout=dropout,
                                  output_length=latent_length,
                                  emb_dropout=0.1)
        
        # self.fc_mu = nn.Linear(dim_model * latent_length, VAE_latent_dim)
        # self.fc_var = nn.Linear(dim_model * latent_length, VAE_latent_dim)

        # self.vae_latent_to_encoder_out = nn.Linear(VAE_latent_dim, dim_model * latent_length)
        self.from_enc_latent_to_memory = nn.Identity() # encoder latent is our latent sapce, in VAE we go from VAE_latent_dim-->dim_model * latent_length, but latent is already that size
        decoder_dim_feedforward = 4 * decoder_input_dim
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(input_dim=decoder_input_dim, d_model=dim_model, nhead=nhead, dim_feedforward=decoder_dim_feedforward) for _ in range(decoder_depth)])

        self.projection_block = nn.Sequential(
            nn.LayerNorm(self.dim_model),
            nn.Linear(self.dim_model, 50)  # Predict next 50 values at each step
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def encode(self, src):
        x, latent = self.encoder(src)
        x = x.view(x.size()[0], -1) # reshape to [b x model_dim * latent_length]

        # mu = self.fc_mu(x)
        # log_var = self.fc_var(x)

        return [x, latent]

    def autoregressive_decode(self, encoder_out, start_tokens, total_len=5000, block_size=50, teacher=None, teacher_forcing_ratio=1.0):
        b = start_tokens.shape[0]
        device = start_tokens.device
        assert total_len % block_size == 0, "total_len must be divisible by block_size"

        num_blocks = total_len // block_size
        decoder_input = start_tokens.unsqueeze(1) #decoder_input: stores the running sequence of previously generated blocks; shape starts as (B, 1, 50) and grows to (B, 2, 50), ..., (B, 100, 50)

        outputs = []

        # VAE sampling
        # mu, log_var = encoder_out[1], encoder_out[2]
        # std = torch.exp(0.5 * log_var)
        # z = mu + std * torch.randn_like(std)
        # memory = self.vae_latent_to_encoder_out(z).view(b, self.latent_length, self.dim_model)

        # latent space becomes memory
        encoder_latent = encoder_out[0]
        # print(f"ecncoder shape: {encoder_latent.shape}")
        memory = self.from_enc_latent_to_memory(encoder_out[0]).view(b, self.latent_length, self.dim_model)
        for i in range(1, num_blocks):
            # USE BELOW IF: only using previous block to predict next
            # last_block = decoder_input[:, -1, :]  # (B, 50)
            # tgt = self.flatten_to_high_dim(last_block).unsqueeze(1)  # (B, 1, dim_model)

            # for layer in self.decoder_layers:
            #     tgt, _, _ = layer(tgt=tgt, memory=memory)

            #  USE BELOW IF: using ALL previous blocks to predict next
            flat_input = decoder_input.view(b, -1, 50)  # (B, i, 50) where i is the number of blocks generated
            tgt = self.flatten_to_high_dim(flat_input)  # (B, i, dim_model) **this is the projection from 50-dim to model dim

            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()  # (i, i) causal mask
            for layer in self.decoder_layers:
                tgt, _, _ = layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask) # (B, i, dim_model) output

            last_hidden = tgt[:, -1, :]  # (B, dim_model) -> utilizes the last-generated block (now attention aware) to generate next 50
            next_block = self.projection_block(last_hidden)  # (B, 50)
            outputs.append(next_block)

            # Probabilistic teacher forcing
            if teacher is not None and i < teacher.shape[1] and torch.rand(1).item() < teacher_forcing_ratio:
                next_input = teacher[:, i, :]  # (B, 50)
            else:
                next_input = next_block

            decoder_input = torch.cat([decoder_input, next_input.unsqueeze(1)], dim=1) # appending next block to decoder input

        return torch.cat(outputs, dim=1)  # (B, 4950)

    def forward(self, src, full_target=None, teacher_forcing_ratio=1.0):
        encoder_out = self.encode(src)

        # Split full_target into blocks of 50
        if full_target is not None:
            blocks = full_target.view(full_target.shape[0], -1, 50)  # (B, 100, 50)
            start_tokens = blocks[:, 0, :]  # first block is the 50 start tokens
            targets = blocks[:, 1:, :]  # the remaining 99 blocks to be predicted, so using first 50 to prefict all 4950
        else:
            start_tokens = torch.ones(src.size(0), 50, device=src.device)
            targets = None

        predictions = self.autoregressive_decode(
            encoder_out,
            start_tokens=start_tokens,
            total_len=5000,
            block_size=50,
            teacher=targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return predictions, encoder_out[1]#, encoder_out[2]