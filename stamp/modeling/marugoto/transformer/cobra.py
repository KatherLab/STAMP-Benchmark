import torch
import torch.nn as nn
import sys
sys.path.append("../")
sys.path.append("../ssl")
sys.path.append("../../")
sys.path.append("../../Mamba2MIL")
#sys.path.append("../../Mamba2MIL/models")
#sys.path.append("../../Mamba2MIL/mamba")
from Mamba2MIL.models.MambaMIL import MambaMIL

class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             #nn.Identity(),
             #nn.Dropout(0.25),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x)

class MambaMILmocoWrap(nn.Module):
    def __init__(self,embed_dim, c_dim,input_dim=1536,layer=4,att_dim=256):
        super().__init__()
        self.embed = nn.ModuleDict({#"512":Embed(512,embed_dim),
                                    "768":Embed(768,embed_dim),
                                    "1024":Embed(1024,embed_dim),
                                  "1280":Embed(1280,embed_dim),
                                  "1536":Embed(1536,embed_dim),})
        #if use_embed:
        input_dim = embed_dim
        self.mamba_mil = MambaMIL(input_dim,embed_dim,n_classes=embed_dim,layer=layer,att_dim=att_dim)
        self.proj = nn.Sequential(
            #nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,4*embed_dim),
            nn.SiLU(),
            #nn.Identity(),
            #nn.Dropout(),
            nn.Linear(4*embed_dim,c_dim),
            #nn.BatchNorm1d(c_dim),
        )

        #self.classifier = nn.Linear(embed_dim,num_classes)
        for param in self.mamba_mil.parameters():
            param.requires_grad = False
        for param in self.mamba_mil.attention.parameters():
            param.requires_grad = True
        for param in self.mamba_mil.classifier.parameters():
            param.requires_grad = True
            
    def forward(self, x,lens,get_tile_embs=False,get_weighted_avg=True):
       
        
        #if self.use_embed:
        embs = self.embed[str(x.shape[-1])](x)
        #else:
        #    embs = x
        if get_tile_embs:
            return self.mamba_mil(embs)[1]
        if get_weighted_avg:
            A = self.mamba_mil(embs,hidden_states=False,return_weighting=True)[1]
            #print(f"{A.shape=}")
            return self.proj(torch.bmm(A,x).squeeze(1))
        logits, _ = self.mamba_mil(embs)
        feats = self.proj(logits.squeeze(1))
        return feats