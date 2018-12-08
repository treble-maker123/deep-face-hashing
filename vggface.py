import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
import multiprocessing
from time import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import *

class VGGFace(nn.Module):
    '''
    '''
    def __init__(self, hash_dim=48, split_num=10, num_classes=530):
        super(VGGFace, self).__init__()
        self.vgg = models.vgg19_bn()

        self.fc1 = nn.Linear(25088, 4092)
        self.do1 = nn.Dropout(p=0.50)
        self.fc2 = nn.Linear(4092, hash_dim)
        self.fc3 = nn.Linear(hash_dim, num_classes)

    def forward(self, X):
        features = self.vgg.features(X) # outputs 25088 features
        flatten = features.view((features.shape[0], -1))

        fc1 = F.relu(self.fc1(flatten))
        codes = torch.tanh(self.fc2(fc1))
        scores = torch.softmax(self.fc3(codes), dim=1)

        return codes, scores

# ==========================
# Hyperparameters
# ==========================

# number of epochs to train
NUM_EPOCHS = 40
# the number of hash bits in the output
HASH_DIM = 48
# the distance to use for calculating precision/recall
HAMM_RADIUS = 2
# top_k closet images to score for mean average precision
TOP_K = 50
# optimizer parameters
OPTIM_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 0.0
}
CUSTOM_PARAMS = {
    "beta": 0.001, # quantization loss regularizer
    "img_size": 224
}
BATCH_SIZE = {
    "train": 64,
    "gallery": 64,
    "val": 64,
    "test": 64
}
LOADER_PARAMS = {
    "num_workers": multiprocessing.cpu_count() - 2,
    # "num_workers": 1
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

data_train = FaceScrubDataset(type="label",
                              mode="train",
                              transform=TRANSFORMS,
                              hash_dim=HASH_DIM)

data_val = FaceScrubDataset(type="label",
                            mode="val",
                            transform=TRANSFORMS,
                            hash_dim=HASH_DIM)

data_test = FaceScrubDataset(type="label",
                             mode="test",
                             transform=TRANSFORMS,
                             hash_dim=HASH_DIM)

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

model_class = VGGFace

def train(model, loader, optim, logger, **kwargs):
    '''
    Train for one epoch.
    '''
    device = kwargs.get("device", torch.device("cpu"))
    print_iter = kwargs.get("print_iter", 40)

    model.to(device=device)
    # set model to train mode
    model.train()
    quant_losses = []
    score_losses = []

    for num_iter, (X, y) in enumerate(loader):
        optim.zero_grad()

        X = X.to(device).float()
        y = y.to(device).long()
        scores, codes = model(X)
        # quantization loss
        quant_loss = CUSTOM_PARAMS['beta'] * (codes.abs() - 1).abs().mean()
        # score error
        score_loss = F.cross_entropy(scores, y)
        # total loss
        loss = quant_loss + score_loss
        loss.backward()
        # apply gradient
        optim.step()
        # save the lossses
        quant_losses.append(quant_loss.item())
        score_losses.append(score_loss.item())

        if (num_iter+1) % print_iter == 0:
            logger.write(
                "iter {} ".format(num_iter+1) +
                "- quant loss: {:.8f}, score loss: {:.8f}"
                    .format(quant_loss, score_loss))

    return sum(quant_losses)/len(quant_losses), \
           sum(score_losses)/len(score_losses)
