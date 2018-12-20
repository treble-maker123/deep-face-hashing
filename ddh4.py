import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import multiprocessing
from time import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import *

class DDH4(nn.Module):
    '''
    # ==========================================================================
    # Discriminative Deep Hashing for Scalable Face Image Retrieval
    # https://www.ijcai.org/proceedings/2017/0315.pdf
    # ==========================================================================

    Introduced distance loss metrics.

    Conv1 = 3x3 kernel, 1 stride, 20 dim (output 31x31)
    Batch
    Pool1 = 2x2 kernel (output 15x15)

    Conv2 = 2x2 kernel, 1 stride, 40 dim (output 14x14)
    Batch
    Pool2 = 2x2 kernel (output 7x7)

    Conv3 = 2x2 kernel, 1 stride, 60 dim (output 6x6)
    Batch
    Pool3 = 2x2 kernel (output 3x3)

    Conv4 = 2x2 kernel, 1 stride, 80 dim (output 2x2)
    Batch

    Merge = 60*3*3 + 80*2*2 = 860

    Split into K groups, let K = 96

    480 face features
    48 groups of 10 features
    48-bits

    # ==========================================================================
    # Simultaneous Feature Learning and Hash Coding with Deep Neural Networks
    # https://arxiv.org/pdf/1504.03410.pdf
    # ==========================================================================
    '''
    def __init__(self, hash_dim=48, split_num=10, num_classes=530):
        super().__init__()
        self.cn1 = nn.Conv2d(3, 20, kernel_size=3)
        nn.init.kaiming_normal_(self.cn1.weight)
        self.bn1 = nn.BatchNorm2d(20)
        self.mp1 = nn.MaxPool2d(2)

        self.cn2 = nn.Conv2d(20, 40, kernel_size=2)
        nn.init.kaiming_normal_(self.cn2.weight)
        self.bn2 = nn.BatchNorm2d(40)
        self.mp2 = nn.MaxPool2d(2)

        self.cn3 = nn.Conv2d(40, 60, kernel_size=2)
        nn.init.kaiming_normal_(self.cn3.weight)
        self.bn3 = nn.BatchNorm2d(60)
        self.mp3 = nn.MaxPool2d(2)

        self.cn4 = nn.Conv2d(60, 80, kernel_size=2)
        nn.init.kaiming_normal_(self.cn4.weight)
        self.bn4 = nn.BatchNorm2d(80)

        # merge layer
        self.mg1 = Merge()
        self.fc1 = nn.Linear(29180, hash_dim*split_num)

        # hash layer
        self.de1 = DivideEncode(hash_dim*split_num, split_num)

        self.fc2 = nn.Linear(hash_dim, num_classes)

    def forward(self, X):
        l1 = self.mp1(F.relu(self.bn1(self.cn1(X))))
        l2 = self.mp2(F.relu(self.bn2(self.cn2(l1))))
        l3 = self.mp3(F.relu(self.bn3(self.cn3(l2))))
        l4 = F.relu(self.bn4(self.cn4(l3)))
        # merge of output from layer 3 and 4
        l5 = self.mg1(l3, l4)
        # face feature layer
        l6 = F.relu(self.fc1(l5))
        # divide and encode
        codes = self.de1(l6)
        scores = self.fc2(codes)
        return codes, scores

class Merge(nn.Module):
    '''
    Implementation of the Merged Layer in,

    Discriminative Deep Hashing for Scalable Face Image Retrieval
    https://www.ijcai.org/proceedings/2017/0315.pdf
    '''
    def __init__(self):
        super().__init__()

    def forward(self, X1, X2):
        X1, X2 = self._flatten(X1), self._flatten(X2)
        return self._merge(X1, X2)

    def _flatten(self, X):
        N = X.shape[0]
        return X.view(N, -1)

    def _merge(self, X1, X2):
        return torch.cat((X1, X2), 1)

class DivideEncode(nn.Module):
    '''
    Implementation of the divide-and-encode module in,

    Simultaneous Feature Learning and Hash Coding with Deep Neural Networks
    https://arxiv.org/pdf/1504.03410.pdf
    '''
    def __init__(self, num_inputs, num_per_group):
        super().__init__()
        assert num_inputs % num_per_group == 0, \
            "num_per_group should be divisible by num_inputs."
        self.num_groups = num_inputs // num_per_group
        self.num_per_group = num_per_group
        weights_dim = (self.num_groups, self.num_per_group)
        self.weights = nn.Parameter(torch.empty(weights_dim))
        nn.init.xavier_normal_(self.weights)

    def forward(self, X):
        X = X.view((-1, self.num_groups, self.num_per_group))
        return X.mul(self.weights).sum(2)

# ==========================
# Hyperparameters
# ==========================

# number of epochs to train
NUM_EPOCHS = 60
# the number of hash bits in the output
HASH_DIM = 48
# the distance to use for calculating precision/recall
HAMM_RADIUS = 2
# top_k closet images to score for mean average precision
TOP_K = 50
# optimizer parameters
OPTIM_PARAMS = {
    "lr": 1e-2,
    "weight_decay":2e-4
}
CUSTOM_PARAMS = {
    "alpha": 1.0, # quantization loss regularizer
    "beta": 1.0, # score loss regularizer
    "gamma": 1.0, # distance loss regularizer
    "mu": 6, # threshold for distance contribution to loss
    "print_iter": 40, # print every n iterations
    "img_size": 128
}
BATCH_SIZE = {
    "train": 256,
    "gallery": 256,
    "val": 256,
    "test": 256
}
LOADER_PARAMS = {
    # "num_workers": 4,
    "num_workers": multiprocessing.cpu_count() - 1,
    "collate_fn": invalid_collate
}

# ==========================
# Setup
# ==========================

# uncomment to reset the data
# undo_create_set("val")
# undo_create_set("test")
# create_set("val")
# create_set("test")
TRANSFORMS = [
    T.Resize((CUSTOM_PARAMS['img_size'], CUSTOM_PARAMS['img_size'])),
    T.ToTensor()
]

DATASET_PARAMS = {
    "align": True,
    "type": "label",
    "transform": TRANSFORMS,
    "hash_dim": HASH_DIM
}

data_train = FaceScrubDataset(mode="train",
                              **DATASET_PARAMS)

data_val = FaceScrubDataset(mode="val",
                            **DATASET_PARAMS)

data_test = FaceScrubDataset(mode="test",
                             **DATASET_PARAMS)

# for training use, shuffling
loader_train = DataLoader(data_train,
                          batch_size=BATCH_SIZE["train"],
                          shuffle=True,
                          **LOADER_PARAMS)

# for use as gallery, no shuffling
loader_gallery = DataLoader(data_train,
                          batch_size=BATCH_SIZE["gallery"],
                          shuffle=False,
                          **LOADER_PARAMS)

loader_val = DataLoader(data_val,
                          batch_size=BATCH_SIZE["val"],
                          shuffle=False,
                          **LOADER_PARAMS)
loader_test = DataLoader(data_test,
                          batch_size=BATCH_SIZE["test"],
                          shuffle=False,
                          **LOADER_PARAMS)

model_class = DDH4
model = model_class(hash_dim=HASH_DIM)
optimizer = optim.Adam(model.parameters(), **OPTIM_PARAMS)

def train(model, loader, optim, logger, **kwargs):
    '''
    Train for one epoch.
    '''
    device = kwargs.get("device", torch.device("cpu"))
    print_iter = kwargs.get("print_iter", CUSTOM_PARAMS['print_iter'])

    model.to(device=device)
    # set model to train mode
    model.train()
    quant_losses = []
    score_losses = []

    for num_iter, (X, y) in enumerate(loader):
        optim.zero_grad()

        X = X.to(device).float()
        y = y.to(device).long()
        codes, scores = model(X)

        with torch.no_grad():
            half_size = len(X) // 2

        C1, C2 = codes[:half_size], codes[half_size:]
        y1, y2 = y[:half_size][None, :], y[half_size:][:, None]
        y1, y2 = y1.repeat(half_size, 1), y2.repeat(1, half_size)

        with torch.no_grad():
            if len(C2) > len(C1):
                C2, y2 = C2[:-1], y2[:-1]

        sim_gt = (y1 == y2).float()
        diff_gt = 1 - sim_gt

        # distance loss
        l2_dist = ((C1[:, None, :] - C2) ** 2 + 1e-8).sum(dim=2).sqrt()
        sim_loss = (sim_gt * l2_dist).mean()
        threshold = torch.max(CUSTOM_PARAMS['mu'] - l2_dist,
                              torch.zeros_like(l2_dist))
        diff_loss = ((1 - sim_gt) * threshold).mean()
        dist_loss = 0.10 * sim_loss + 0.90 * diff_loss
        # quantization loss
        quant_loss = (codes.abs() - 1).abs().mean()
        # score error
        score_loss = F.cross_entropy(scores, y)
        # # total loss
        loss = CUSTOM_PARAMS['alpha'] * quant_loss + \
               CUSTOM_PARAMS['beta'] * score_loss + \
               CUSTOM_PARAMS['gamma'] * dist_loss
        loss.backward()
        # apply gradient
        optim.step()
        # save the lossses
        quant_losses.append(quant_loss.item())
        score_losses.append(score_loss.item())

        if (num_iter+1) % print_iter == 0:
            logger.write(
                "iter {} ".format(num_iter+1) +
                "- quant loss: {:.4f}, score loss: {:.4f}, sim loss: {:.4f}, diff loss: {:.4f}"
                    .format(quant_loss.item(), score_loss.item(), sim_loss.item(), diff_loss.item()))
