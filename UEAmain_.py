import time
import os
import numpy as np
import seaborn as sns
import pywt
import matplotlib.pyplot as plt
#from math import *
import pandas as pd
from tqdm import tqdm
import argparse
from IPython import embed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from datasets.uea_dataset import UCR_Dataset
from new_furi import furi_utils
from new_furi import furi_more as Fur
import pickle
def load_pkl(path):
    pickle_file = open(path,'rb')
    obj=pickle.load(pickle_file)
    pickle_file.close()
    return obj
class WDHA(nn.Module):
    def __init__(self, input_shape, n_class,wave_base_ml,wide=15,num_waves=18,dim=None,V_dim=32):

        # Furi_bases:(n,L',wide);L'=w*dim
        #data_len=input_shape[2]
        #dim0=input_shape[1]
        super(WDHA, self).__init__()
        self.input_shape=input_shape

        self.embed = nn.LSTM(input_shape[1], dim, batch_first=True,bidirectional=True)

        self.wide=wide
        self.wave_base = wave_base_ml
        self.conv_forbasescale= nn.ModuleList([])
        self.conv_forpoint_paraall= nn.ModuleList([])
        self.conv_forpoint_para0 =nn.ModuleList([])
        for i in self.wave_base:

            self.conv_forbasescale.append(nn.Conv2d(1, 1, (1,self.wide),padding = (0, int((self.wide - 1) / 2))))


        num_waves=len(self.wave_base)






        self.bn1 = nn.BatchNorm1d(dim * 2 * (num_waves ))  # 44？
        self.conv2 = nn.Conv1d(dim*2*(num_waves), 128, 3)#

        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, V_dim, 5)
        # (n,  128)

        self.bn3 = nn.BatchNorm1d(V_dim)
        self.conv4 = nn.Conv1d(V_dim, 16, 3)
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(5, 16, 3)
        self.bn5 = nn.BatchNorm1d(16)
        self.dwt=furi_utils.DTW_1d_dim((V_dim,input_shape[2]-6))


        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(in_features=16, out_features=n_class)
        self.dropout = nn.Dropout(p=0.5)


    def count_conv_parameters(self):
        conv_layers = [self.conv1]
        param_num = 0
        for conv_layer in conv_layers:
            for p in conv_layer.parameters():
                param_num += p.numel()
        return param_num

    def forward(self, x):#x:(batch_size, data_channels, data_len);bi_lstm
        x=x.transpose(1,2)
        out, (h_n, c_n)=self.embed(x)
        Output = out.transpose(1, 2)
        # Output:(batch,  hidden_size * num_directions,seq_len)
        out_all=Output[:,:,0]+Output[:,:,-1]#(batch,  dim)





        Outputembed=Output.unsqueeze(1)#x:(batch_size,1, dim0* num_directions, data_len)

        outlist=[]

        for ii in range(len(self.wave_base)):
            scales = self.conv_forbasescale[ii](Outputembed)#(Output)#Output: (N, 1, H_{out}(dim), W_{out})
            scales = F.relu(scales)
            Furi_bases = self.wave_base[ii].return_base(scales)  # return_basefu,return_basefu_0
            # (N,dim,length,wide)
            Attw = torch.sum(Furi_bases,dim=-1,keepdim=True)

            out = furi_utils.Furi_conv(Outputembed, Furi_bases, wide=self.wide)#(Output)  # (n, 1, dim, w)
            sqrs = torch.sqrt(((scales / (1 / self.wide)).trunc() + 1) / (self.wide + 2) * 2)

            out = out / sqrs
            outlist.append(out)
        wave_conv=torch.cat(outlist, 1)  # (n, num_waves, dim, w)




        IM_wave=wave_conv
        IM_wave1d=IM_wave.view(IM_wave.size(0), -1, IM_wave.size(-1))
        #(n, (num_waves)* dim, w)
        out = self.bn1(IM_wave1d)

        out = F.relu(self.conv2(out))#w1=w-(kernalsize-1)-1+1=w-kernalsize+1
        out = self.bn2(out)##(n, 128, w1)

        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out_back=out

        out = F.relu(self.conv4(out))

        out = self.bn4(out)#w3=w2-kernalsize+1=w2-2
        if (out_back.size(2) > 8):
            outdwt4=self.dwt.foward_4_FULL_fix_all(out_back)
            outdwt4 =F.relu(self.conv5(outdwt4))
            outdwt4 = self.bn5(outdwt4)
            out=torch.cat([out,outdwt4],-1)

        out = self.avgpool(out)
        out = out.squeeze_(-1)#(n,  128)

        out = self.fc1(out)
        return out







def train_model(train_loader, model, optimizer, writer, epoch, step_cnt):
    model.train()
#     print("Start Training...")  
    total = 0
    correct = 0
    losses = []
    
    for _, batch in enumerate(train_loader):     
        optimizer.zero_grad()
        batch_X = batch['feature'].to(device)
        batch_y = batch['label'].reshape(-1).to(device)
        criterion = nn.CrossEntropyLoss()        
        logits = model(batch_X)
        # embed()
        loss = criterion(logits, batch_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
        
        step_cnt['train'] += 1
        batch_acc = pred.eq(batch_y).sum().item() / batch_y.size(0)
        batch_loss = loss.item()

        
        writer.add_scalar('Loss/train_batch', batch_loss, step_cnt['train'])
        writer.add_scalar('Accuracy/train_batch', batch_acc, step_cnt['train'])
    
    losses = np.array(losses)
    train_loss = losses.mean()  
    train_acc = correct / total
    
    writer.add_scalar('Loss/train', train_loss, step_cnt['train'])
    writer.add_scalar('Accuracy/train', train_acc, step_cnt['train'])
        
    return train_loss, train_acc

def test_model(test_loader, model, writer, epoch,write_interval=30):
    model.eval()
    total = 0
    correct = 0
    losses = []
#     print("Start Testing...")
    for _, batch in enumerate(test_loader):  ###
        batch_X = batch['feature'].to(device)
        batch_y = batch['label'].reshape(-1).to(device)
        


        criterion = nn.CrossEntropyLoss() 
        logits = model(batch_X)
        # embed()
        loss = criterion(logits, batch_y)
        # print("loss:", loss)
        losses.append(loss.item())
        _, pred = logits.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
        
    losses = np.array(losses)
    test_loss = losses.mean()  
    test_acc = correct / total
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    
    return test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset_name', type=str, default='ERing')
    parser.add_argument('--exp_dir', type=str, default='uae_exp')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--V_dim', type=int, default=32)
    parser.add_argument('--wide', type=int, default=15)
    parser.add_argument('--wave_used', type=int, default=18)
    parser.add_argument('--embed_dim', type=int, default=None)


    args = parser.parse_args()
    print('V_dim',args.V_dim,'wide',args.wide,'wave_used',args.wave_used,'embed_dim',args.embed_dim)



    args.exp_dir ='UEA'
    model_name = 'WDHA'+'V_dim'+str(args.V_dim)+'wide'+str(args.wide)+'wave_used'+str(args.wave_used)+'embed_dim'+str(args.embed_dim)

    
    batch_size = args.batch_size
    epochs = args.epochs
    dataset_name = args.dataset_name
    exp_dir = args.exp_dir

    runs_dir = os.path.join(exp_dir, 'runs', dataset_name)
    weights_dir = os.path.join(exp_dir, 'weights', dataset_name)
    logs_dir = os.path.join(exp_dir, 'logs', dataset_name)

    model_save_path = os.path.join(weights_dir, model_name + '.pth')
    if (os.path.exists(model_save_path)):
        print('model existed')
        import sys
        sys.exit(1)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")


    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    multidim = 0
    train_dataset = UCR_Dataset(train=True,
                                root_dir='Multivariate_ts',
                                dataset_name=dataset_name)
    test_dataset = UCR_Dataset(train=False,
                               root_dir='Multivariate_ts',
                               dataset_name=dataset_name, multidim=multidim, maxlength=train_dataset.lenmax)
    


    batch_size = min(batch_size, train_dataset.features.shape[0] // 10)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers=4)

    data_len = train_dataset.features.shape[2]
    data_channels = train_dataset.features.shape[1]
    n_class = len(set(train_dataset.labels.flatten().tolist()))
    input_shape = (batch_size, data_channels, data_len)

    writer = SummaryWriter(log_dir=runs_dir)

    path = 'furi_15_parameters'
    p_array_list=load_pkl(path)
    furi_list = [Fur.fu_one_0,Fur.fu_one_1,Fur.fu_one_2,Fur.fu_one_3,Fur.fu_one_4,Fur.fu_one_5,Fur.fu_one_6,Fur.fu_one_7,Fur.fu_one_8,Fur.fu_one_9,Fur.fu_one_10,Fur.fu_one_11,Fur.fu_one_12,Fur.fu_one_13,Fur.fu_one_14]
    wave_names=[ 'db2', 'db3','db4'
     , 'haar',  'sym2', 'sym3', 'sym4'
    , 'coif1', 'coif2', 'coif3'
    , 'morl', 'shan'
    , 'gaus1', 'gaus2', 'gaus3', 'mexh'
        , 'cgau1', 'cgau2', 'cgau3'
    , 'fbsp', 'cmor', 'dmey']
    wide0=args.wide
    wave_used = args.wave_used
    # furi_list=furi_list[0:furi_used]
    Wave_base_l = nn.ModuleList(
        [furi_utils.Wave_base(furi_list, p_array_list[i], wide=wide0, grad_para_mul=0.01) for i in
         range(wave_used)])


    if (args.embed_dim==None):
        args.embed_dim=data_channels
    model=WDHA(input_shape, n_class,Wave_base_l,wide=wide0,
                                num_waves=len(Wave_base_l),dim=args.embed_dim,V_dim=args.V_dim).to(device)




    #optimizer = optim.Adam(model.parameters(), lr=0.00003, eps=0.001)
    step_cnt = {'train': 0, 'test': 0}
    max_test_acc = 0
    train_logs = []

    for epoch in range(epochs):
        lrn = 0.01 / (1 + epochs // 60)#0.005
        optimizer = optim.Adam(model.parameters(), lr=lrn, eps=0.001)  # lr=0.00003
        print("*" * 50)
        print("Training on dataset:", dataset_name)
        print("Epoch:", epoch)
        train_loss, train_acc = train_model(train_loader, 
                                            model, 
                                            optimizer, 
                                            writer,
                                            epoch,
                                            step_cnt)
        print("Train Loss: %.3f, Train Accuracy: %.3f" % (train_loss, train_acc) )
        test_loss, test_acc = test_model(test_loader, 
                                        model, 
                                        writer,

                                        epoch)
        print("Test Loss: %.3f, Test Accuracy: %.3f" % (test_loss, test_acc))
        if max_test_acc < test_acc:
            max_test_acc = test_acc
            #torch.save(model, model_save_path)


        print("Now max test acc:", max_test_acc)
        
        now_epoch_result = {
                            'epoch':epoch,
                            'train_loss':train_loss,
                            'train_acc':train_acc,
                            'test_loss':test_loss,
                            'test_acc':test_acc,
                            'max_test_acc':max_test_acc
                            }
        train_logs.append(now_epoch_result)

        train_logs_df = pd.DataFrame(train_logs)
        train_logs_path = os.path.join(logs_dir, model_name + '.csv')
        train_logs_df.to_csv(train_logs_path)

    print("Final max test acc:", max_test_acc)

    # test_load_model = torch.load(model_save_path)








