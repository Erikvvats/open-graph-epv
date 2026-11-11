"""Small graph transformer with optional, player-aligned trajectory attention.

The trajectory encoder shares its Q/K/V/output weights with the first spatial
block. All three conditions therefore have exactly the same parameter count.
This is a capacity-controlled model, not the parent's larger temporal model.
"""
import math
import torch
from torch import nn
from .data import FEATURES


class Attention(nn.Module):
    def __init__(self,hidden,heads):
        super().__init__()
        self.heads=heads;self.width=hidden//heads
        self.qkv=nn.Linear(hidden,3*hidden);self.out=nn.Linear(hidden,hidden)

    def forward(self,x,mask,bias=None,last_query=False):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,self.width).permute(2,0,3,1,4)
        if last_query:q=q[:,:,-1:]
        scores=q@k.transpose(-1,-2)/math.sqrt(self.width)
        if bias is not None:scores=scores+bias
        # Dummy key only for fully masked trajectories; outputs remain masked.
        valid=mask.clone();valid[:,0]|=~valid.any(1)
        scores=scores.masked_fill(~valid[:,None,None,:],-1e9)
        out=(scores.softmax(-1)@v).transpose(1,2).reshape(b,1 if last_query else n,d)
        query_mask=mask[:,-1:] if last_query else mask
        return self.out(out)*query_mask[...,None]


class SpatialBlock(nn.Module):
    def __init__(self,hidden,heads):
        super().__init__()
        self.norm=nn.LayerNorm(hidden);self.attention=Attention(hidden,heads)
        self.edge_bias=nn.Linear(4,heads)
        self.ff_norm=nn.LayerNorm(hidden)
        self.ff=nn.Sequential(nn.Linear(hidden,2*hidden),nn.SiLU(),nn.Linear(2*hidden,hidden))

    def forward(self,h,mask,edges):
        h=h+self.attention(self.norm(h),mask,self.edge_bias(edges).permute(0,3,1,2))
        return (h+self.ff(self.ff_norm(h)))*mask[...,None]


class AttentionEPV(nn.Module):
    def __init__(self,hidden=40,layers=2,heads=4,mode='snapshot',offsets=(-20,-10,-5,0)):
        super().__init__()
        if hidden%heads or mode not in ('snapshot','history','repeat'):raise ValueError('Invalid transformer configuration')
        if not offsets or offsets[-1]!=0 or any(t>0 for t in offsets) or sorted(set(offsets))!=list(offsets):
            raise ValueError('History offsets must be distinct, sorted and end at the present')
        self.config=dict(hidden=hidden,layers=layers,heads=heads,mode=mode,offsets=list(offsets))
        self.mode=mode;self.offsets=offsets
        self.encoder=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.SiLU())
        self.time_encoder=nn.Linear(1,hidden,bias=False)
        self.blocks=nn.ModuleList([SpatialBlock(hidden,heads) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(3*hidden+2,hidden),nn.SiLU(),nn.Linear(hidden,1))

    def forward(self,x,mask):
        if x.ndim!=4 or mask.shape!=x.shape[:3] or x.shape[1]!=len(self.offsets):
            raise ValueError('Expected player-aligned x[B,H,N,F] and mask[B,H,N]')
        mask=mask.bool();x=torch.where(mask[...,None],x,torch.zeros_like(x))
        if self.mode=='snapshot':
            x=x[:,-1:];mask=mask[:,-1:];offsets=[0]
        else:
            offsets=self.offsets
            if self.mode=='repeat':
                # Both temporal conditions receive the same past visibility for
                # current players; only real-history coordinates/motion differ.
                mask=mask&mask[:,-1:]
                x=x[:,-1:].expand(-1,len(offsets),-1,-1)
                x=torch.where(mask[...,None],x,torch.zeros_like(x))
        b,t,n,_=x.shape;current=x[:,-1];current_mask=mask[:,-1]
        times=torch.as_tensor(offsets,dtype=x.dtype,device=x.device).view(1,t,1,1)/10
        h=(self.encoder(x)+self.time_encoder(times))*mask[...,None]
        trajectories=h.permute(0,2,1,3).reshape(b*n,t,-1)
        visible=mask.permute(0,2,1).reshape(b*n,t)
        first=self.blocks[0]
        history=first.attention(first.norm(trajectories),visible,last_query=True).reshape(b,n,-1)
        h=(h[:,-1]+history)*current_mask[...,None]
        relative=current[:,None,:,:2]-current[:,:,None,:2]
        distance=relative.square().sum(-1,keepdim=True).sqrt()
        same=(current[:,:,None,4]*current[:,None,:,4]+current[:,:,None,5]*current[:,None,:,5])[...,None]
        edges=torch.cat([relative,distance,same],-1)
        for block in self.blocks:h=block(h,current_mask,edges)
        pools=[];counts=[]
        for feature in (4,5,6):
            weight=current[...,feature]*current_mask
            pools.append((h*weight[...,None]).sum(1)/weight.sum(1).clamp_min(1)[:,None])
            counts.append(weight.sum(1,keepdim=True)/11)
        return self.head(torch.cat(pools+counts[:2],-1)).squeeze(-1).tanh()

    def symmetric(self,x,mask):
        mirror=x.clone();mirror[...,1]*=-1;mirror[...,3]*=-1
        return .5*(self(x,mask)+self(mirror,mask))
