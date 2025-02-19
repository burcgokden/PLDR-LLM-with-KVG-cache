''''
Power law attention implementation for PLDR-LLM v510G with pytorch and KV-cache and G-cache.
v510G removes the deep residual layers and custom weight/biases that learn A, A_LM and G_LM
G_LM is provided as a predefined value during model initialization.
'''

import torch
from torch import nn
import torch.nn.functional as F


class plga_layer(nn.Module):
    '''
    Power law graph attention layer implementation.
    '''
    def __init__(self,  F_hidden, Gcache, device=None, **kwargs):
        '''
        Args:
            F_hidden: hidden layer shape used in layer weight creation. For multi-head plga this is depth.
            Gcache: Predefined G_LM.
            device: device(cpu or gpu) to load tensors.
        '''

        super().__init__(**kwargs)
        self.F_hidden=F_hidden
        self.device=device
        self.avAp=Gcache


    def cg_align_one(self, Hin, Hkt, mask=None):
        '''
        Alignment model for calculating E with elements eij
        Args:
            Hin: query
            Hkt: transpose of key
            mask: padding or lookahead mask
        Returns:
            E: attention weights applied on value
        '''
        WHiWHj = torch.matmul(Hin, self.avAp) #[batch_size, num_head, seq_lenq, depth]
        Ep=torch.matmul(WHiWHj, Hkt) #[batch_size, num_head, seq_lenq, seq_lenk]

        #scale attention with square root of depth
        dk=torch.tensor(self.F_hidden).to(torch.float32)
        Ep=Ep/torch.sqrt(dk)

        #apply mask and softmax
        E= Ep + (mask * -1e9) if mask is not None else Ep
        E=F.softmax(E, dim=-1)

        return E 
    
    def cg_align_head(self, Hin, Hk, Hv, mask=None):
        '''
        Method for linear propagation of attention weights over values.
        '''

        Hkt = torch.permute(Hk, [0, 1, 3, 2])  # (batch_size, num_head, depth, seq_lenk)

        Eout=self.cg_align_one(Hin, Hkt, mask=mask)

        Hout = torch.matmul(Eout, Hv) #[batch_size, num_heads, seq_lenq ,d_model]

        return Hout


    def forward(self, inputs, **kwargs):
        '''
        execute the forward propagation
        inputs[0] = query = Hin
        inputs[1] = key = Hk
        inputs[2] = value = Hv
        inputs[3] = mask
        '''
        Hin, Hk, Hv, mask=inputs
        H_next = self.cg_align_head(Hin, Hk, Hv, mask=mask)
        return H_next

def iSwiGLU(x):
    '''SwiGLU activation function with weights W,V equal to identity matrix and no bias.'''
    gate=F.silu(x)
    out=torch.mul(x, gate)
    return out

