import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
#import opts
import time
import math

import os



def D_ler_0(x):

    return torch.zeros_like(x)

def D_ler_1(x):

    return torch.ones_like(x)

def D_ler_2(x):
    x_=3*x

    return x_
def D_ler_3(x):
    x_=15/2*x*x-3/2

    return x_



def ler_0(x):
    if(type(x)==np.ndarray):
        return np.ones_like(x)
    else:
        return torch.ones_like(x)



def ler_1(x):

    return x

def ler_2(x):
    x_=(3*x*x-1)/2

    return x_

def ler_3(x):
    x_=(5*x*x*x-3*x)/2

    return x_
def ler_4(x):
    x_=(35*x*x*x*x-30*x*x+3)/8

    return x_
def ler_5(x):
    x_=(63*x.pow(5)-70*x.pow(3)+15*x)/8

    return x_
def ler_6(x):
    x_=(231*x.pow(6)-315*x.pow(4)+105*x.pow(2)-5)/16

    return x_
def ler_7(x):
    x_=(429*x.pow(7)-693*x.pow(5)+315*x.pow(3)-35*x)/16

    return x_
def ler_8(x):
    x_=(6435* x.pow(8)-12012*x.pow(6)+6930*x.pow(4)-1260*x.pow(2)+35)/128

    return x_
def ler_9(x):
    x_=(12155* x.pow(9)-25740*x.pow(7)+18018*x.pow(5)-4620*x.pow(3)+315*x.pow(1))/128

    return x_
def ler_10(x):
    x_=(46189* x.pow(10)-109395*x.pow(8)+90090*x.pow(6)-30030*x.pow(4)+3465*x.pow(2)-63)/256

    return x_



def fu_one(No):
    No = No + 1
    def fu_one_(x):  # No=0,1,2
          # No=1,2,3
        x = 3.1415926535898 * x
        if (No % 2 == 0):
            y = torch.sin((No // 2) * x)
        elif (No % 2 == 1):
            y = torch.cos((No // 2) * x)
        else:
            print('wrong fu_one(No')
        return y
    return fu_one_




class Grad_switch(Function):
    @staticmethod
    def forward(self, Fout,mul):

        self.save_for_backward(Fout,mul)
        return Fout

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):

        Fout,mul = self.saved_tensors
        grad_mul=torch.zeros_like(mul)

        return grad_output * mul,grad_mul




def f_Grad_switch(Fout,mul):

    return Grad_switch()(Fout,mul)


class Wave_base(nn.Module):#foward:where function;many 0;scale parameter
    #form modulelist
    def __init__(self,furi_list,p_array,wide=8,grad_para_mul=0.01):#p_array:(len(furi_list),)
        super(Wave_base, self).__init__()
        self.furi_list=furi_list
        self.n = len(furi_list)
        #c=p_array.astype('float32')
        self.p_array_t = nn.Parameter(torch.from_numpy(p_array.astype('float32')))
        self.p_array_t.requires_grad = False
        self.para_train=nn.Parameter(torch.zeros_like(self.p_array_t,requires_grad=True))
        x = np.linspace(-1, 1, wide).astype('float32')##(-1.5, 1.5, wide)

        self.base_x=nn.Parameter(torch.from_numpy(x))
        self.base_x.requires_grad = False
        self.one_base=nn.Parameter(torch.ones_like(self.base_x,requires_grad=False))
        self.zero_base = nn.Parameter(torch.zeros_like(self.base_x, requires_grad=False))
        self.grad_para_mul=nn.Parameter(Variable(torch.ones(1), requires_grad=False)*grad_para_mul)

    def   return_base(self, scale):#scale:(N, 1, H_{out}(dim), W_{out})   self.base_x:(wide,)
        para_train_=Grad_switch.apply(self.para_train, self.grad_para_mul)#f_Grad_switch(self.para_train, self.grad_para_mul)
        base_x_more=self.base_x.unsqueeze(0)#(1，wide)
        base_x_more = base_x_more.unsqueeze(0)#(1，1，wide)
        base_x_more = base_x_more.unsqueeze(0)  # (1，1，1，wide)
        scale=scale.transpose(1,2).transpose(2,3)#(N,dim,length,1)
        base_x_s=base_x_more*scale#(N,dim,length,wide)
        area=torch.where(((base_x_s>-1) * (base_x_s<1)),self.one_base,self.zero_base).to(self.base_x.device)#and base_x_s<1
        #area.requires_grad = False
        area=area.detach()
        furi=self.furi_list[0](base_x_s)*(self.p_array_t[0]+para_train_[0])
        for i in range(1,self.n):
            furi=furi+self.furi_list[i](base_x_s)*(self.p_array_t[i]+para_train_[i])
        F_base=furi*area
        return F_base#F_base：(N,dim,length,wide)

    def return_base_fu(self, scale,point_para):#scale:(N, 1, H_{out}(dim), W_{out})#W_{out}=length   self.base_x:(wide,)
        #point_para:(n,input_shape[1]*2,self.num_furi,w)
        para_train_=Grad_switch.apply(self.para_train, self.grad_para_mul)
        base_x_more=self.base_x.unsqueeze(0)#(1，wide)
        base_x_more = base_x_more.unsqueeze(0)#(1，1，wide)
        base_x_more = base_x_more.unsqueeze(0)  # (1，1，1，wide)
        scale=scale.transpose(1,2).transpose(2,3)#(N,dim,length,1)
        base_x_s=base_x_more*scale#(N,dim,length,wide)
        area=torch.where(((base_x_s>-1) * (base_x_s<1)),self.one_base,self.zero_base).to(self.base_x.device)
        area=area.detach()
        point_para=point_para.transpose(2,3)#(N,dim,length,len(furi_list))
        point_para=point_para.unsqueeze(-1)#(N,dim,length,len(furi_list),1)
        furi=self.furi_list[0](base_x_s)*(self.p_array_t[0]+para_train_[0]+point_para[:,:,:,0,:])
        for i in range(1,self.n):

            furi=furi+self.furi_list[i](base_x_s)*(self.p_array_t[i]+para_train_[i]+point_para[:,:,:,i,:])
        F_base=furi*area
        return F_base
    def return_base_fu_0(self, scale,point_para):#scale:(N, 1, H_{out}(dim), W_{out})   self.base_x:(wide,)
        para_train_=f_Grad_switch(self.para_train, self.grad_para_mul)
        base_x_more=self.base_x.unsqueeze(0)#(1，wide)
        base_x_more = base_x_more.unsqueeze(0)#(1，1，wide)
        base_x_more = base_x_more.unsqueeze(0)  # (1，1，1，wide)
        scale=scale.transpose(1,2).transpose(2,3)#(N,dim,length,1)
        base_x_s=base_x_more*scale#(N,dim,length,wide)
        area=torch.where(base_x_s>-1 and base_x_s<1,self.one_base,self.zero_base).to(self.base_x.device)
        area.requires_grad = False
        furi=self.furi_list[0](base_x_s)*(self.p_array_t[0]+para_train_+point_para[:,:,0])
        for i in range(1,self.n):
            furi=furi+self.furi_list[i](base_x_s)*(self.p_array_t[i]+para_train_)
        F_base=furi*area
        return F_base
    def return_base_back(self, scale):#scale:(N,length) self.base_x:(wide,)
        para_train_=f_Grad_switch(self.para_train, self.grad_mul)
        base_x_more=self.base_x.unsqueeze(0)#(1，wide)
        base_x_more = base_x_more.unsqueeze(0)#(1，1，wide)
        scale=scale.unsqueeze(-1)#(N,length,1)
        base_x_s=base_x_more*scale#(N,length,wide)
        area=torch.where(base_x_s>-1 and base_x_s<1,self.one_base,self.zero_base).to(self.base_x.device)
        area.requires_grad = False
        furi=self.furi_list[0](base_x_s)*(self.p_array_t[0]+para_train_[0])
        for i in range(1,self.n):
            furi=furi+self.furi_list[i](base_x_s)*(self.p_array_t[i]+para_train_[i])
        F_base=furi*area
        return F_base#F_base：(N,length,wide)

def Furi_conv(input,Furi_bases,wide=8):#input:（ n, c, h, w）,c=1,h=dim,w=length
    #Furi_bases:(N,dim,length,wide)to:(n,L',wide);L'=w*dim||
    #Furi_bases:torch.Size([10,  2, 1460, 8])
    #input:torch.Size([10, 1, 2, 1460])

    Furi_bases=Furi_bases.view(Furi_bases.size(0),  -1, Furi_bases.size(3))
    batch=input.size(0)
    w=input.size(3)

    #Unfold=torch.nn.Unfold( (1, wide),padding=1)
    inp_unf=torch.nn.functional.unfold(input,(1, wide),padding=(0,int((wide-1)/2)))#(N,wide,L);(N,C×∏(kernel_size),L),C×∏(kernel_size)=wide;L=w*dim
    Furi_bases=Furi_bases#10, 2920, 9#.transpose(1, 2)#10, 9, 2920
    inp_unf=inp_unf.transpose(1, 2)#10, 2920, 9
    out_unf=torch.sum(Furi_bases*inp_unf,dim=2)
    #out_unf = inp_unf.transpose(1, 2).dot(Furi_bases.transpose(1, 2)).transpose(1, 2)#(N,L)
    #Furi_bases:torch.Size([10, 2920, 9])inp_unf:torch.Size([10, 9, 2920])
    out=out_unf.view(batch, 1, -1, w)
    return out#out:(n, 1, dim, w)c=1,dim2=dim*dim*w,w=length
import torch.nn.functional as F



class DTW_1d_dim(nn.Module):#foward:where function;many 0;scale parameter
    #form modulelist
    def __init__(self,input_shape,dim=5):#p_array:(len(furi_list),)
        super(DTW_1d_dim, self).__init__()
        #input_shape=(dim',length')/ ###or (N,length,dim)
        self.dim=dim
        self.DWT_all=nn.Parameter(torch.rand((input_shape[0],input_shape[1],dim)))
        self.DWT=None

    def foward_4_FULL_fix_all(self,input):
        dis_list=[]
        for i in range(self.dim):
            self.DWT=self.DWT_all[ :, :,i]
            dis_list.append(self.foward_4_FULL_fix_each(input))
        TotalD = torch.cat(dis_list, 1)
        return TotalD

    def foward_4_FULL_fix_each(self,input,visual=0):#input_shape=(N,dim',length')
        I0=input[:,:,:-3]
        I1=input[:,:,1:-2]
        I2=input[:,:,2:-1]
        I3 = input[:, :, 3:]
        D0 = self.DWT[ :, :-3].unsqueeze(0)
        D1 = self.DWT[ :, 1:-2].unsqueeze(0)
        D2 = self.DWT[ :, 2:-1].unsqueeze(0)
        D3 = self.DWT[ :, 3:].unsqueeze(0)

        d1=(I0-D0).pow(2)+(I1-D1).pow(2)+(I2-D2).pow(2)+(I3-D3).pow(2)#(N,dim,length'-4)
        d1=d1.unsqueeze(-1)

        d2 = (I0-D0).pow(2)+(I0 - D1).pow(2) + (I1 - D2).pow(2) + (I2 - D3).pow(2)+ (I3 - D3).pow(2)#(N,dim,length'-4)
        d2 = d2.unsqueeze(-1)
        d3 = (I0 - D0).pow(2) + (I0 - D1).pow(2) + (I0 - D2).pow(2) \
             + (I1 - D3).pow(2)+ (I2 - D3).pow(2)+ (I3 - D3).pow(2)  # (N,dim,length'-4)
        d3 = d3.unsqueeze(-1)
        d4 = (I0 - D0).pow(2) + (I1 - D1).pow(2) + (I1 - D2).pow(2) + (I2 - D3).pow(2) \
             + (I3 - D3).pow(2)  # (N,dim,length'-4)
        d4 = d4.unsqueeze(-1)

        d5 = (I0 - D0).pow(2) + (I0 - D1).pow(2) + (I1 - D2).pow(2) + (I2 - D2).pow(2) \
             + (I3 - D3).pow(2)  # (N,dim,length'-4)
        d5 = d5.unsqueeze(-1)

        d2_ = (D0 - I0).pow(2) + (D0 - I1).pow(2) + (D1 - I2).pow(2) + (D2 - I3).pow(2) + (D3 - I3).pow(
            2)  # (N,dim,length'-4)
        d2_ = d2_.unsqueeze(-1)
        d3_ = (D0 - I0).pow(2) + (D0 - I1).pow(2) + (D0 - I2).pow(2) \
              + (D1 - I3).pow(2) + (D2 - I3).pow(2) + (D3 - I3).pow(2)  # (N,dim,length'-4)
        d3_ = d3_.unsqueeze(-1)
        d4_ = (D0 - I0).pow(2) + (D1 - I1).pow(2) + (D1 - I2).pow(2) + (D2 - I3).pow(2) \
             + (D3 - I3).pow(2)  # (N,dim,length'-4)
        d4_ = d4_.unsqueeze(-1)

        d5_ = (D0 - I0).pow(2) + (D0 - I1).pow(2) + (D1 - I2).pow(2) + (D2 - I2).pow(2) \
             + (D3 - I3).pow(2)  # (N,dim,length'-4)
        d5_ = d5_.unsqueeze(-1)


        TotalD=torch.cat([d1,d2,d3,d4,d5,d2_,d3_,d4_,d5_],-1)#(N,dim,length'-4,paths_num)
        TotalD=torch.sum(TotalD,dim=1,keepdim=True)#(N,1,length'-4,paths_num)


        Di=F.softmax(-1*TotalD, dim=-1)#(N,1,length'-4,1)
        output=torch.sum(TotalD*Di,dim=-1)#(N,1,length'-4)


        return output




