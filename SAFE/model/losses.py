from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def triplet_loss(anchor, positive, negative, alpha=0.2):
 
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.clamp(pos_dist - neg_dist + alpha, min=0.0)
    
    return loss.mean()

class Sim_ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.25):
        super(Sim_ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos = nn.CosineSimilarity(dim=1)
        similarity = cos(output1, output2)
        # print(similarity.shape)
        loss = torch.mean((1-label) * torch.pow(similarity, 2) +
                          label * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
        return loss
