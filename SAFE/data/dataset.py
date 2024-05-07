import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import collections
import glob 
from tqdm import tqdm
import random
import re
from sklearn.preprocessing import StandardScaler
import json


continous_features = ['gdmj', 'ydmj', 'ldmj', 'cdmj', 'sdmj', 'nyssjsydmj', 'jtysydmj', 'czjsydmj', 'cunzjsydmj', 'sgssydmj', 'ldsy', 'wlyd', 'tchd', 
                    'gdggmj', 'gdxtzs', 'yn_2017', 'yphx', 'gc', 'pd', 'rksl']
categorial_features = ['gdmjdj', 'ydmjdj', 'ldmjdj', 'cdmjdj', 'sdmjdj', 'nyssjsydmjdj', 'jtysydmjdj', 'czjsydmjdj', 'cunzjsydmjdj', 'sgssydmjdj',
                     'ldsydj', 'wlyddj', 'trzd', 'trlx', 'nyjgyqyds', 'jzzfwzjl']

config = json.load(open('./config.json', 'r'))

with open(config['data_dir'] + '/geohashes.txt', 'r') as f:
    geodata = f.readlines()
    geodata = [geo.strip() for geo in geodata]

geohash_to_ind = {geo:i for i, geo in enumerate(geodata)}

class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature, bias):

        self.dicts = []
        self.num_feature = num_feature
        self.latitude_indexes = []
        self.longtitude_indexes = []
        self.area_indexes = []
        self.bias = bias
        with open(config['data_dir'] + '/geohash_to_index.txt', 'r') as f:
            spatial_index = f.readlines()
            wall = spatial_index[0].strip().split(',')
            self.wall = [int(w) for w in wall]
            self.wall_lat = max(self.wall[0], self.wall[2], self.wall[4]) + 2*self.bias
            self.wall_lon = max(self.wall[1], self.wall[3], self.wall[5]) + 2*self.bias
            spatial_index = spatial_index[1:]
            spatial_index = [sindex.strip().split(',') for sindex in spatial_index]
            self.spatial_index = {sindex[0]:sindex[1:] for sindex in spatial_index}
        

    def build(self, datafile, categorial_features, cutoff=0):
        data = pd.read_csv(datafile)
        length = 2100
        value = {
            'tchd':0,
            'trzd':1000,
            'gdggmj':0,
            'gdxtzs':0,
            'rksl': 0
            }
        data = data.fillna(value=value)

        self.geohash = [geohash_to_ind[geo] for geo in data['geohash'][:length]]
        self.map = []
        for k in range(self.num_feature):
            attribute = categorial_features[k]
            col_data = list(set(data[attribute]))
            cate_to_num = collections.defaultdict(float)
            for ind, cd in enumerate(col_data):
                cate_to_num[cd] = ind
            cate_to_num['unk'] = ind + 1
            self.map.append(cate_to_num)
        #len(data)
        for i in range(length):
            line = data.loc[i]
            tmp = []
            for j in range(0, self.num_feature):
                val = line[categorial_features[j]]
                tmp.append(self.map[j][val])
            self.dicts.append(tmp)
        self.final_data = torch.tensor(self.dicts)

        # 构建spatital数组
        for geo in data['geohash'][:length]:
            self.latitude_indexes.append(int(self.spatial_index[geo][1]) + self.bias)
            self.longtitude_indexes.append(int(self.spatial_index[geo][2]) + self.bias)
            self.area_indexes.append(int(self.spatial_index[geo][0]))
        self.latitude_indexes = torch.tensor(self.latitude_indexes)
        self.longtitude_indexes = torch.tensor(self.longtitude_indexes)
        self.area_indexes = torch.tensor(self.area_indexes)

        self.final_spatial_data = torch.zeros((length, self.num_feature, 2*self.bias, 2*self.bias))
        for j in range(0, self.num_feature):
            spatital_data = torch.zeros((3, self.wall_lat, self.wall_lon))
            spatital_data[self.area_indexes, self.latitude_indexes, self.longtitude_indexes] = torch.tensor(list(data[categorial_features[j]][:length])).float()

            for i in range(length):
                self.final_spatial_data[i, j, :, :] = spatital_data[self.area_indexes[i], self.latitude_indexes[i]-self.bias:self.latitude_indexes[i]+self.bias, self.longtitude_indexes[i]-self.bias:self.longtitude_indexes[i]+self.bias]



class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature, bias):
        self.num_feature = num_feature
        self.final_data = []
        self.latitude_indexes = []
        self.longtitude_indexes = []
        self.area_indexes = []
        self.bias = bias
        with open(config['data_dir'] + '/geohash_to_index.txt', 'r') as f:
            spatial_index = f.readlines()
            wall = spatial_index[0].strip().split(',')
            self.wall = [int(w) for w in wall]
            self.wall_lat = max(self.wall[0], self.wall[2], self.wall[4]) + 2*self.bias
            self.wall_lon = max(self.wall[1], self.wall[3], self.wall[5]) + 2*self.bias
            spatial_index = spatial_index[1:]
            spatial_index = [sindex.strip().split(',') for sindex in spatial_index]
            self.spatial_index = {sindex[0]:sindex[1:] for sindex in spatial_index}

    def build(self, datafile, continous_features):
        data = pd.read_csv(datafile)
        length = 2100
        #len(data)
        value = {
            'tchd':0,
            'trzd':1000,
            'gdggmj':0,
            'gdxtzs':0,
            'rksl': 0
            }
        data = data.fillna(value=value)

        self.kzstmj = np.array(list(data['kzstmj'][:length]))
        for i in range(length):
            line = data.loc[i]
            tmp = []
            for j in range(0, self.num_feature):
                val = line[continous_features[j]]
                if val == '':
                    val = 0.0
                else:
                    val = float(val)
                tmp.append(val)
            self.final_data.append(tmp)
        self.final_data = np.array(self.final_data)
        scaler = StandardScaler()
        self.final_data = scaler.fit_transform(self.final_data)

        # 构建spatital数组
        for geo in data['geohash'][:length]:
            self.latitude_indexes.append(int(self.spatial_index[geo][1]) + self.bias)
            self.longtitude_indexes.append(int(self.spatial_index[geo][2]) + self.bias)
            self.area_indexes.append(int(self.spatial_index[geo][0]))
        self.latitude_indexes = torch.tensor(self.latitude_indexes)
        self.longtitude_indexes = torch.tensor(self.longtitude_indexes)
        self.area_indexes = torch.tensor(self.area_indexes)

        self.final_spatial_data = torch.zeros((length, self.num_feature, 2*self.bias, 2*self.bias))
        for j in range(0, self.num_feature):
            spatital_data = torch.zeros((3, self.wall_lat, self.wall_lon))
            spatital_data[self.area_indexes, self.latitude_indexes, self.longtitude_indexes] = torch.tensor(list(data[continous_features[j]][:length])).float()

            for i in range(length):
                self.final_spatial_data[i, j, :, :] = spatital_data[self.area_indexes[i], self.latitude_indexes[i]-self.bias:self.latitude_indexes[i]+self.bias, self.longtitude_indexes[i]-self.bias:self.longtitude_indexes[i]+self.bias]

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, root, train=True, mode='train', year=2019):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train
        self.mode = mode
        self.continuous_num = len(continous_features)
        dataset_mode = config['dataset_mode']
        if mode == 'train':
            self.file = config['data_dir'] + f'/zx_kzsttj{year}_{dataset_mode}_train.csv'
        elif mode == 'valid':
            self.file = config['data_dir'] + f'/zx_kzsttj{year}_{dataset_mode}_valid.csv'
        else:
            self.file = config['data_dir'] + f'/zx_kzsttj{year}_{dataset_mode}_test.csv'

        # for cnn
        Continuousfeature_cnn = ContinuousFeatureGenerator(len(continous_features), 16)
        Continuousfeature_cnn.build(self.file, continous_features)
        self.iniContinuousdata = Continuousfeature_cnn.final_data

        Categoryfeature_cnn = CategoryDictGenerator(len(categorial_features), 16)
        Categoryfeature_cnn.build(self.file, categorial_features)
        self.iniCategorydata = Categoryfeature_cnn.final_data

        self.geohash_ind = Categoryfeature_cnn.geohash
        self.kzstmj = Continuousfeature_cnn.kzstmj

        self.spatial_data_cnn = torch.cat([Continuousfeature_cnn.final_spatial_data, Categoryfeature_cnn.final_spatial_data], dim=1)
        print(f'for cnn: {self.spatial_data_cnn.shape}')


        # for ind, (previous, now) in enumerate(zip(self.iniContinuousdata[0], self.iniContinuousdata[0])):
        tmp = np.zeros((self.iniContinuousdata.shape[0], self.iniContinuousdata.shape[1]+1+self.iniCategorydata.shape[1]))
        tmp[:, :self.iniContinuousdata.shape[1]] = self.iniContinuousdata
        tmp[:, self.iniContinuousdata.shape[1]:-1] = self.iniCategorydata
        tmp[:, -1][self.kzstmj > 7266] = 1

        self.iniContinuousdata = tmp

        # if self.train:
        self.train_data = tmp[:, :-1]
        self.target = tmp[:, -1]
        indices = np.nonzero(self.target == 1)[0]
        self.anchor_index = random.choice(indices.tolist())
        self.anchor = self.train_data[self.anchor_index, :].copy()

        # for anchor
        # index of continous features are zero
        Xi_coutinous = np.zeros_like(self.anchor[:self.continuous_num])
        Xi_categorial = self.anchor[self.continuous_num:]
        self.anchor_Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
        
        # value of categorial features are one (one hot features)
        Xv_categorial = np.ones_like(self.anchor[self.continuous_num:])
        Xv_coutinous = self.anchor[:self.continuous_num]
        self.anchor_Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
        self.anchor_cnn = self.spatial_data_cnn[self.anchor_index]
        self.anchor_gnn = self.spatial_data_cnn[self.anchor_index]
        print(f'Loading {self.mode} dataset done-----Number of {self.mode} dataset: {len(self.train_data)}')
    
    def __getitem__(self, idx):
        # if self.train:
        dataI, targetI = self.train_data[idx, :], self.target[idx]
        # index of continous features are zero
        Xi_coutinous = np.zeros_like(dataI[:self.continuous_num])
        Xi_categorial = dataI[self.continuous_num:]
        Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
        
        # value of categorial features are one (one hot features)
        Xv_categorial = np.ones_like(dataI[self.continuous_num:])
        Xv_coutinous = dataI[:self.continuous_num]
        Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))

        if self.mode == 'train':
            return Xi, Xv, targetI, self.spatial_data_cnn[idx], self.spatial_data_cnn[idx], self.anchor_Xi, self.anchor_Xv, self.anchor_cnn, self.anchor_gnn
        else:
            return Xi, Xv, targetI, self.spatial_data_cnn[idx], self.spatial_data_cnn[idx], self.geohash_ind[idx]


    def __len__(self):
        return len(self.train_data)

    def _check_exists(self):
        return os.path.exists(self.root)
