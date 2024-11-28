import torch
import torch.nn as nn
import sys

from .mamba2 import Mamba2Enc
from .abmil import BatchedABMIL
import torch.nn.functional as F
from einops import rearrange
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024,dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             nn.Dropout(dropout) if dropout else nn.Identity(),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x) 

class Cobra(nn.Module):
    def __init__(self,embed_dim, c_dim,layer=4,dropout=0.25,num_heads=8,freeze_base=True):
        super().__init__()
        
        self.embed = nn.ModuleDict({#"512":Embed(512,embed_dim),
                                   "768":Embed(768,embed_dim),
                                   "1024":Embed(1024,embed_dim),
                                   "1280":Embed(1280,embed_dim),
                                    "1536":Embed(1536,embed_dim),})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=layer,dropout=dropout)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,4*embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(4*embed_dim,c_dim),
            nn.BatchNorm1d(c_dim),
        )

        
        self.num_heads = num_heads
        #self.abmil = BatchedABMIL(embed_dim,hidden_dim=att_dim,dropout=dropout,n_classes=embed_dim)
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),hidden_dim=int(embed_dim/num_heads),dropout=dropout,n_classes=1) for _ in range(self.num_heads)])
        
        if freeze_base:
            for param in self.embed.parameters():
                param.requires_grad = False
            for param in self.mamba_enc.parameters():
                param.requires_grad = False
        
    def forward(self, x):

        logits = self.embed[str(x.shape[-1])](x)

        h = self.norm(self.mamba_enc(logits))#+self.norm(logits)
        #print(f"{logits.shape=}")
        
        
        if self.num_heads > 1:
            h_ = rearrange(h, 'b t (e c) -> b t e c',c=self.num_heads)

            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(h_[:, :, :, i], return_raw_attention = True) # , return_raw_attention = True
                attention.append(processed_attention)
                
            A = torch.stack(attention, dim=-1)

            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1)
            A = torch.transpose(A,2,1)
            A = F.softmax(A, dim=-1) 
        else: 
            A = self.attn[0](h)
        
        h = torch.bmm(A,x).squeeze(1)
        feats = self.proj(h)
        return feats, h
