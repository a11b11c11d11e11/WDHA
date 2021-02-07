import pywt
import numpy as np
import torch
from new_furi import furi_utils
import matplotlib.pyplot as plt
def get_wavelet_bases(wavelet='gaus1', level=20):
    now_wavelet = None
    phi, psi, x = [None, None, None]
    if wavelet in pywt.wavelist(kind='continuous'):
        now_wavelet = pywt.ContinuousWavelet(wavelet)
        psi, x = now_wavelet.wavefun(level=level)
    else:

        now_wavelet = pywt.Wavelet(wavelet)
        phi, psi, x = now_wavelet.wavefun(level=level)
    return psi,x

def get_furi_para(psi,furi_list):#numpy
    n=psi.size(0)#n=psi.shape[0]
    x=torch.from_numpy(np.linspace(-1, 1, n))
    list=[]
    for function in furi_list:
        param=torch.sum(psi*function(x))/n#param=np.sum(psi*function(x))/n
        list.append(param.item())#list.append(param)
    p_array=np.array(list)
    return p_array

def show_furi_plot(p_array,furi_list,input_num=100):
    n=input_num
    x0=np.linspace(-1, 1, n)
    x = torch.from_numpy(np.linspace(-1, 1, n))

    furi = np.linspace(0, 0, n)
    out_list=[]
    for i in range(len(furi_list)):
        function=furi_list[i]
        param=p_array[i]
        furi=furi+function(x).numpy()*param
        out_list.append(furi)
    import matplotlib.pyplot as plt
    for out in out_list:
        plt.plot(x0, out)
        plt.show()
    return out_list
import pickle
def save_pkl(path,obj):
    pickle_file = open(path,'wb')
    pickle.dump(obj, pickle_file)
    pickle_file.close()
def load_pkl(path):
    pickle_file = open(path,'rb')
    obj=pickle.load(pickle_file)
    pickle_file.close()
    return obj

if __name__ == '__main__':
    furi_list = [furi_utils.fu_one(i) for i in range(15)]
    p_array_list=[]
    for text in [ 'db2', 'db3','db4'
     , 'haar',  'sym2', 'sym3', 'sym4'
    , 'coif1', 'coif2', 'coif3'
    , 'morl', 'shan'
    , 'gaus1', 'gaus2', 'gaus3', 'mexh'
        , 'cgau1', 'cgau2', 'cgau3'
    , 'fbsp', 'cmor', 'dmey']:
        psi,xx = get_wavelet_bases(wavelet=text, level=20)
        psi = torch.from_numpy(psi)
        p_array = get_furi_para(psi, furi_list)
        p_array_list.append(p_array)

    path='furi_15_parameters'
    save_pkl(path, p_array_list)



