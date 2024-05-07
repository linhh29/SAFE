# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""
from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score,f1_score,recall_score, roc_auc_score
from time import time
from .Classifier import create_cosNorm_model, create_lws_model
from .mobilenet import mobilenet
from sklearn.metrics import confusion_matrix
from .losses import ContrastiveLoss, triplet_loss, Sim_ContrastiveLoss
from .vig import vig_ti_224_gelu
from .TabTransformer import TabTransformer_block
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self):
        super(BalancedSoftmax, self).__init__()
        freq = torch.tensor([322958, 1758])
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class SAFE(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, seed, num_categories, num_continuous, config=None,batch_size=1000, embedding_size=4, hidden_dims=[32, 32], num_classes=3, dropout=[0.5, 0.5], 
                 use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs: 
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.model = TabTransformer_block(
                categories = feature_sizes[num_continuous:],      # tuple containing the number of unique values within each category
                num_continuous = num_continuous,                # number of continuous values
                dim = 32,                           # dimension, paper set at 32
                dim_out = 2,                        # binary prediction, but could be anything
                depth = 6,                          # depth, paper recommended 6
                heads = 8,                          # heads, paper recommends 8
                attn_dropout = 0.1,                 # post-attention dropout
                ff_dropout = 0.1,                   # feed forward dropout
                mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
                mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            )
        self.config = config
        self.seed = seed
        self.continuous_num = num_continuous
        self.categories_num = num_categories
        self.field_size = len(feature_sizes)
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.batch_size = batch_size
        self.dtype = torch.long

        # +512
        self.out1 = nn.Linear(532+512, 2)
        # +512
        self.out2 = create_cosNorm_model(in_dims=532+512, out_dims=2)

        self.img_model = mobilenet(class_num=72, channel=self.field_size, kernel_size=3, padding=1)
        self.gcn = vig_ti_224_gelu()

    def forward(self, Xi, Xv, img_cnn, img_gcn, mode):
        X_cate = Xi[:, self.continuous_num:]
        X_cont = Xv[:, :self.continuous_num]
        # print(X_cate.device)
        # print(X_cont.device)
        pred = self.model(X_cate.squeeze(), X_cont)

        cnn_feature = self.img_model(img_cnn)
        gcn_out, gcn_feature = self.gcn(img_gcn)

        now_len = pred.shape[0]
        if mode == 'train':
            deepfm_out = pred[: now_len//2]
            anchor_deepfm_out = pred[now_len//2:]
            anchor_cnn_feature = cnn_feature[now_len//2:]
            cnn_feature = cnn_feature[:now_len//2]
            anchor_gcn_feature = gcn_feature[now_len//2:]
            gcn_feature = gcn_feature[:now_len//2]

            # cnn_feature
            x = torch.cat([deepfm_out, cnn_feature], dim=1)
            # + gcn_out[:now_len//2]
            total_sum = self.out1(x) + self.out2(x) + gcn_out[:now_len//2]
            anchor_out = torch.cat([anchor_gcn_feature], dim=1)
            feature_out = torch.cat([gcn_feature], dim=1)
            return total_sum, anchor_out, feature_out
            # return total_sum, deepfm_out, anchor_deepfm_out, cnn_feature, anchor_cnn_feature, gcn_feature, anchor_gcn_feature
        else:
            # , cnn_feature
            x = torch.cat([pred, cnn_feature], dim=1)
            # + gcn_out
            total_sum = self.out1(x) + self.out2(x)+ gcn_out
            return total_sum


    def fit(self, loader_train, loader_val, loader_test, optimizer, config, epochs=100, verbose=False, print_every=100):
        model = self.train().to(device=self.device)
        criterion_classifier = BalancedSoftmax()
        # criterion_contrastive = ContrastiveLoss()
        criterion_contrastive = Sim_ContrastiveLoss()
        # criterion_classifier = nn.CrossEntropyLoss()

        max_f1 = 0
        for e in range(epochs):  
            sys.stdout.flush()   
            for t, (xi, xv, y, img_cnn, img_gcn, anchor_xi, anchor_xv, anchor_cnn, anchor_gnn) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.long)
                img_cnn = img_cnn.to(device=self.device, dtype=torch.float)
                img_gcn = img_gcn.to(device=self.device, dtype=torch.float)
                anchor_cnn = anchor_cnn.to(device=self.device, dtype=torch.float)
                anchor_gnn = anchor_gnn.to(device=self.device, dtype=torch.float)

                anchor_xi = anchor_xi.to(device=self.device, dtype=self.dtype)
                anchor_xv = anchor_xv.to(device=self.device, dtype=torch.float)

                xi = torch.cat([xi, anchor_xi], dim=0)
                xv = torch.cat([xv, anchor_xv], dim=0)
                img_cnn = torch.cat([img_cnn, anchor_cnn], dim=0)
                img_gcn = torch.cat([img_gcn, anchor_gnn], dim=0)

                total, anchor_out, feature_out = model(xi, xv, img_cnn, img_gcn, mode='train')
                # total, deepfm_out, anchor_deepfm_out, cnn_feature, anchor_cnn_feature, gcn_feature, anchor_gcn_feature = model(xi, xv, img_cnn, img_gcn, mode='train')
                #   + criterion_contrastive(feature_out, anchor_out, y) 
                # closs = (criterion_contrastive(deepfm_out, anchor_deepfm_out, y)+criterion_contrastive(cnn_feature, anchor_cnn_feature, y)+criterion_contrastive(gcn_feature, anchor_gcn_feature, y))/3
                loss = criterion_classifier(total, y) + criterion_contrastive(feature_out, anchor_out, y)      
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                    print('--------------------valid---------------------')
                    f_valid = self.check_accuracy(loader_val, model, f'Epoch{e} Iteration{t}')
                    print('--------------------test---------------------')
                    f_test = self.check_accuracy(loader_test, model, f'Epoch{e} Iteration{t}')
                    print()

                    if f_test >= max_f1:
                        max_f1 = f_test
                        out_model_pth = config['result_dir'] + f'/best_{self.seed}.pt'
                        self.save(model, out_model_pth)
        dataset_mode = config['dataset_mode']
        with open(f'./ours.txt_{dataset_mode}', 'a') as f:
            f.write(str(max_f1)+'\n')

    
    def check_accuracy(self, loader, model, name_string):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)
        geohashes = torch.tensor([]).to(self.device)

        with open(self.config['data_dir'] + '/geohashes.txt', 'r') as f:
            geodata = f.readlines()
            geodata = [geo.strip() for geo in geodata]

        ind_to_geohash = {i: geo for i, geo in enumerate(geodata)}

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y, img_cnn, img_gcn, geohash_inds in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
                img_cnn = img_cnn.to(device=self.device, dtype=torch.float)
                img_gcn = img_gcn.to(device=self.device, dtype=torch.float)  
                

                geohash_inds = geohash_inds.to(device=self.device, dtype=torch.int)
                total = model(xi, xv, img_cnn, img_gcn, 'valid')
                
                preds = torch.argmax(F.softmax(total, dim=1), dim=1)
                # print(preds)
                # print(y)
                # print(preds.shape)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                
                y_true = torch.cat([y_true, y], dim=0)
                y_pred = torch.cat([y_pred, preds], dim=0)
                y_score = torch.cat([y_score, F.softmax(total, dim=1)[:, 1]], dim=0)
                geohashes = torch.cat([geohashes, geohash_inds], dim=0)

            # print(y_true)
            # print(y_pred)
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            y_score = y_score.cpu().numpy()
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f = f1_score(y_true, y_pred, average='macro')
            AUC = roc_auc_score(y_true, y_score)
            print('Got %d / %d Precision (%.3f) Recall (%.3f) F1 (%.3f) AUC (%.3f)' % (num_correct, num_samples,precision, recall, f, AUC))
            confusion = confusion_matrix(y_true, y_pred)
            print(f'Confusion Matrix:\n{confusion}')

            # print(y_pred.shape)
            # print(y_true.shape)
            # print(geohashes.shape)
            return f

    def save(self, model, path):
        torch.save(model.state_dict(), path)



                        
