''''
Power law attention implementation for PLDR-LLM v510 with pytorch and KV-cache and G-cache.
'''

import torch
from torch import nn
import torch.nn.functional as F


class plga_layer(nn.Module):
    '''
    Power law graph attention layer implementation.
    '''
    def __init__(self, F_hidden, F_heads, a_init=None, W_init=None, 
                 b_init=None, pw_init=None, device=None, **kwargs):
        '''
        Args:
            F_hidden: hidden layer shape used in layer weight creation. For multi-head plga this is depth.
            F_heads: Number of attention heads.
            a_init: initializer for learnable coupling coefficients.
            W_init: initializer for weight used in attention model.
            b_init: initializer for bias values.
            pw_init: initializer for power weights.
            device: device(cpu or gpu) to load tensors.
        '''

        super().__init__(**kwargs)
        self.F_hidden=F_hidden
        self.F_heads=F_heads
        self.W_initializer = W_init if W_init is not None else nn.init.xavier_normal_ 
        self.pw_initializer = pw_init if pw_init is not None else nn.init.xavier_normal_ 
        self.a_initializer=a_init if a_init is not None else nn.init.xavier_normal_ 
        self.b_initializer=b_init if b_init is not None else nn.init.zeros_ 
        self.device=device

        self.build_weights()


    def cg_align_one(self, Hin, Hkt, A, a_vec, ba, W, b, pw, mask=None, Gcache=None):
        '''
        Alignment model for calculating E with elements eij
        Args:
            Hin: query
            Hkt: transpose of key
            A: metric tensor instance
            a_vec: learnable coupling coefficients.
            ba: bias for coupling coeffients
            W: weights appliead on metric tensor before AdjActivation
            b: bias applied on metric tensor before AdjActivation
            pw:power values applied on metric tensor
            mask: padding or lookahead mask
            Gcache: A list containing cache of [A_LM, G_LM]
        Returns:
            A list of:
                E: attention weights applied on value
                AW: metric tensor after AdjActivation is applied, A_LM
                pw: learned power values
                a_vec: learned coupling coefficients
                ba: bias for coupling coefficients
                avAp: Energy curvature tensor, G_LM
                Ep: Energy-curvature tensor before mask is applied
        '''

        if Gcache is None:
            We = torch.tile(W[None, :,:,:], [Hin.size()[0], 1, 1, 1])  # [batch_size, num_head, depth, depth]
            a_vece = torch.tile(a_vec[None, :,:,:], [Hin.size()[0], 1, 1, 1])  # [batch_size, num_head, depth, depth]
            AdjActivation=iSwiGLU
            epsilonAdj=1e-9

            #make metric tensor positive definite
            AW=AdjActivation(torch.matmul(We,A)+b)+epsilonAdj

            #find energy curvature tensor and attention weights
            pwe = torch.tile(pw[None, :,:,:], [Hin.size()[0], 1, 1, 1])  # [batch_size, num_head,  depth, depth]
            Ap=torch.pow(AW, pwe)
            avAp=torch.matmul(a_vece, Ap)+ba # [batch_size, num_head,  depth, depth]
        else:
            AW=Gcache[0]
            avAp=Gcache[1]

        WHiWHj = torch.matmul(Hin, avAp) #[batch_size, num_head, seq_lenq, depth]
        Ep=torch.matmul(WHiWHj, Hkt) #[batch_size, num_head, seq_lenq, seq_lenk]

        #scale attention with square root of depth
        dk=torch.tensor(self.F_hidden).to(torch.float32)
        Ep=Ep/torch.sqrt(dk)

        #apply mask and softmax
        E= Ep + (mask * -1e9) if mask is not None else Ep
        E=F.softmax(E, dim=-1)

        return E, [AW, pw, a_vec, ba, avAp, Ep]
    
    def cg_align_head(self, Hin, Hk, Hv, A, mask=None, Gcache=None):
        '''
        Method for linear propagation of attention weights over values.
        '''

        Hkt = torch.permute(Hk, [0, 1, 3, 2])  # (batch_size, num_head, depth, seq_lenk)

        Eout, att_weights=self.cg_align_one(Hin, Hkt, A, 
                                            a_vec=self.alst,
                                            ba=self.balst,
                                            W=self.Wlst,
                                            b=self.blst,
                                            pw=self.pwlst, 
                                            mask=mask,
                                            Gcache=Gcache)

        Hout = torch.matmul(Eout, Hv) #[batch_size, num_heads, seq_lenq ,d_model]

        return Hout, att_weights



    def build_weights(self):
        '''
        Used to initialize learnable parameters for the layer:
        W: weights to apply on metric tensor
        b: bias to apply on metric tensor
        a: coupling coefficients for power law attention
        ba: bias for power law attention.
        pw: power weights for power law attention
        '''

        weight_shape=[self.F_heads, self.F_hidden, self.F_hidden] #[num_heads, depth, depth]

        add_weight_Wpart= torch.empty(weight_shape, dtype=torch.float32, device=self.device)
        add_weight_Wpart=self.W_initializer(add_weight_Wpart)

        add_weight_bpart=torch.empty(weight_shape, dtype=torch.float32, device=self.device)
        add_weight_bpart=self.b_initializer(add_weight_bpart)

        add_weight_pwpart=torch.empty(weight_shape, dtype=torch.float32, device=self.device)
        add_weight_pwpart=self.pw_initializer(add_weight_pwpart)

        add_weight_apart = torch.empty(weight_shape, dtype=torch.float32, device=self.device)
        add_weight_apart=self.a_initializer(add_weight_apart)

        add_weight_bapart=torch.empty(weight_shape, dtype=torch.float32, device=self.device)
        add_weight_bapart=self.b_initializer(add_weight_bapart)

        self.Wlst = nn.Parameter(add_weight_Wpart, requires_grad=True)
        self.blst = nn.Parameter(add_weight_bpart, requires_grad=True) 
        self.pwlst = nn.Parameter(add_weight_pwpart, requires_grad=True)  
        self.alst = nn.Parameter(add_weight_apart, requires_grad=True) 
        self.balst = nn.Parameter(add_weight_bapart, requires_grad=True) 


    def forward(self, inputs, Gcache=None, **kwargs):
        '''
        execute the forward propagation
        inputs[0] = query = Hin
        inputs[1] = key = Hk
        inputs[2] = value = Hv
        inputs[3] = metric tensor = A
        inputs[4] = mask
        '''
        Hin, Hk, Hv, A, mask=inputs
        H_next, att_weights = self.cg_align_head(Hin, Hk, Hv, A, mask=mask, Gcache=Gcache)
        return H_next, att_weights

def iSwiGLU(x):
    '''SwiGLU activation function with weights W,V equal to identity matrix and no bias.'''
    gate=F.silu(x)
    out=torch.mul(x, gate)
    return out

