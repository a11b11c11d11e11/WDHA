import os
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import os
import sktime

def numpy_fillna(data,maxlength=0):
    list=[]
    lenmax=0
    for i in data:
        for ii in i:
            len=ii.shape[0]
            lenmax=max(lenmax,len)
    lenmax = max(lenmax, maxlength)
    for i in data:
        listi=[]
        for ii in i:
            len = ii.shape[0]
            listi.append(np.pad(ii, (0,lenmax-len), 'constant', constant_values=(0,0)))

        list.append(listi)
    return list,lenmax

def numpy_transform(npa,part,indim=2,outdim=1,keep=3):
    slice_list=[]
    now=0
    if(indim==2):
        total=npa.shape[2]
        e=total//part
        for i in range(part):
            if(i==0):
                slice_list.append(npa[:,:,now:now+e+keep*2+1])
            elif(i==(part-1)):
                slice_list.append(npa[:, :, -1-keep * 2-1-e:-1])
            else:
                slice_list.append(npa[:, :, now-keep:now + keep+e+1])
            now=now+e
    if (indim == 1):
        total = npa.shape[1]
        e = total // part
        for i in range(part):
            if (i == 0):
                slice_list.append(npa[:, now:now+e + keep * 2 + 1, :])
            elif (i == (part - 1)):
                slice_list.append(npa[:, -1 - keep * 2-1-e:-1, :])
            else:
                slice_list.append(npa[:, now - keep:now + keep+e+1, :])
            now = now + e

    npa2=np.concatenate(slice_list, axis=outdim)
    return npa2


def load_dataset(train, root_dir, dataset_name,maxlength=0):
    raw_dataset = None
    if train:
        dataset_path = os.path.join(root_dir, dataset_name, 
                                    dataset_name+'_TRAIN.ts')
        #raw_dataset = load_from_tsfile_to_dataframe(dataset_path)
        npy_path0 = os.path.join(root_dir, dataset_name,
                                    dataset_name + '_TRAIN_0.npy')
        npy_path1 = os.path.join(root_dir, dataset_name,
                                 dataset_name + '_TRAIN_1.npy')
    else:
        dataset_path = os.path.join(root_dir, dataset_name, 
                                    dataset_name+'_TEST.ts')

        npy_path0 = os.path.join(root_dir, dataset_name,
                                dataset_name + '_TEST_0.npy')
        npy_path1 = os.path.join(root_dir, dataset_name,
                                 dataset_name + '_TEST_1.npy')
    if (os.path.exists(npy_path0)):
        features=np.load(npy_path0)
        labels=np.load(npy_path1)
        return features, labels,features.shape[2]
    else:
        raw_dataset = load_from_tsfile_to_dataframe(dataset_path)

    #list(raw_dataset[0].iloc[i][ii].value for ii in range(raw_dataset[0].shape[1]))
    c=raw_dataset[0].iloc[0][0].values
    f=list(list(raw_dataset[0].iloc[i][ii].values for ii in range(raw_dataset[0].shape[1])) for i in range(raw_dataset[0].shape[0]))
    f2,lenmax=numpy_fillna(f,maxlength=maxlength)
    #f=raw_dataset[0].iloc[0][0]
    features=np.array(f2)
    #featuresc=numpy_fillna(features)
    features=features.astype(np.float32)
    # raw_dataset = raw_dataset.astype(np.float32)
    # features = raw_dataset[:, 1:]#raw_dataset[0]
    features = np.nan_to_num(features)
    labels = raw_dataset[1]#labels = raw_dataset[1].astype(np.float32)#raw_dataset[1].astype(np.float32)
    # embed()

    class_list = list(set(labels.flatten().tolist()))
    class_list.sort()

    labelsnew=[]
    for i in range(labels.shape[0]):
        labelsnew.append(class_list.index(labels[i]) * 1.0)
    # embed()
    labels=np.array(labelsnew)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    le.transform(labels)
    np.save(npy_path0,features)
    np.save(npy_path1, labels)

    return features, labels,lenmax

class UCR_Dataset(Dataset):
    def __init__(self, train=True, root_dir=None, 
                 dataset_name=None, transform=None,multidim=0,
                 maxlength=0,add_dim=None,reshape=None,n_transform=None):
        if train == True:
            # load all train
            self.features, self.labels,self.lenmax = load_dataset(train, root_dir, dataset_name,maxlength=maxlength)
            #self.features = np.expand_dims(self.features, 1)
            self.labels = self.labels.astype(np.long)

            if(add_dim!=None):
                self.features=np.pad(self.features, add_dim, 'constant')

            if (reshape != None):
                self.features = np.reshape(self.features, reshape)
            if (multidim > 0):
                self.features = np.repeat(self.features, multidim, axis=1)
            if (n_transform!= None):
                self.features =numpy_transform(self.features,
            n_transform[0],n_transform[1],n_transform[2],n_transform[3])
                #numpy_transform=(part,indim=2,outdim=1,keep=3)


            print("Train shape:", self.features.shape, self.labels.shape)
        else:
            # load all train
            self.features, self.labels,lenmax = load_dataset(train, root_dir, dataset_name,maxlength=maxlength)
            #self.features = np.expand_dims(self.features, 1)

            if (add_dim != None):
                self.features = np.pad(self.features, add_dim, 'constant')

            if (reshape != None):
                self.features = np.reshape(self.features, reshape)
            if (multidim > 0):
                self.features = np.repeat(self.features, multidim, axis=1)
            if (n_transform!= None):
                self.features =numpy_transform(self.features,
            n_transform[0],n_transform[1],n_transform[2],n_transform[3])
            self.labels = self.labels.astype(np.long)
            print("Test shape:", self.features.shape, self.labels.shape)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'feature': self.features[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

