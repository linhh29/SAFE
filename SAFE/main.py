import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import random
from data.dataset import CriteoDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from model.SAFE import SAFE
import json

# 
seeds = [20, 178, 1637, 34, 590, 1000, 80,168, 367, 491]
for seed in seeds:
     print()
     print('------------------------------------------------')
     print(f'random seed: {seed}')
     config = json.load(open('./config.json', 'r'))
     print(config)

     def data_collator(data_list):
          input_id = data_list[0]
          print(len(input_id))
          return data_list


     def setup_seed(seed):
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          np.random.seed(seed)
          random.seed(seed)
          torch.backends.cudnn.deterministic = True

     setup_seed(seed)

     # 900000 items for training, 10000 items for valid, of all 1000000 items
     batch_size = 1000

     # load data
     train_data = CriteoDataset(config['data_dir'], train=True, year=2019)
     loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
     val_data = CriteoDataset(config['data_dir'], train=True, mode='valid', year=2019)
     loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
     test_data = CriteoDataset(config['data_dir'], train=False, mode='test', year=2019)
     loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
     feature_sizes = np.loadtxt(config['data_dir'] + '/feature_sizes_2019.txt', delimiter=',')
     feature_sizes = [int(x) for x in feature_sizes]
     print(feature_sizes)
     num_cate = 0
     num_cont = 0
     for num in feature_sizes:
          if num > 100:
               num_cont += 1
          else:
               num_cate += 1

     model = SAFE(feature_sizes, seed=seed, num_categories=num_cate, num_continuous=num_cont, config=config, batch_size=batch_size, use_cuda=True)
     optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
     model.fit(loader_train, loader_val, loader_test, optimizer, config, epochs=10, verbose=True)

     del batch_size, train_data, loader_train, val_data, loader_val, test_data, loader_test, feature_sizes, model, optimizer, num_cate, num_cont
