import os
import uuid
import multiprocessing
from functools import reduce
from time import time, strftime
from datetime import datetime

import cv2

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt

from pdb import set_trace

from ddh import *
from dataset import *
from logger import *

# reset the data
undo_create_set("val")
undo_create_set("test")
create_set("val")
create_set("test")

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")


num_epochs = 2100

optimizer_params = {
    "lr": 1e-3,
    "weight_decay": 2e-4
}
custom_params = {
    "beta": 1.0 # regularization to guide bits to either 1 or -1
}

# how many bits to map the images to with the network
hash_dim = 48

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((32, 32)),
    T.ToTensor()
])

# build the dataset

data_train = FaceScrubDataset(type="hash_label",
                              mode="train",
                              transform=transform,
                              hash_dim=hash_dim)
data_val = FaceScrubDataset(type="hash_label",
                            mode="val",
                            transform=transform,
                            hash_dim=hash_dim)
data_test = FaceScrubDataset(type="hash_label",
                             mode="test",
                             transform=transform,
                             hash_dim=hash_dim)

# setting up the data loader
batch_size = {
    "train": 256,
    "val": 256,
    "test": 256
}

loader_params = {
    "shuffle": True,
    "num_workers": multiprocessing.cpu_count() - 2
}

loader_train = DataLoader(data_train,
                          batch_size=batch_size["train"],
                          **loader_params)
loader_val = DataLoader(data_val,
                          batch_size=batch_size["val"],
                          **loader_params)
loader_test = DataLoader(data_test,
                          batch_size=batch_size["test"],
                          **loader_params)

# components
model = DiscriminativeDeepHashing(hash_dim=hash_dim)
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), **optimizer_params)

run_id = uuid.uuid4().hex.upper()[0:6]
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = now + "+" + run_id
mkdir(os.getcwd() + "/models")
checkpoint_path = os.getcwd() + "/models/{}_best_weights.pt" \
                    .format(file_name)

# training code
train_losses = []
train_acc = []
val_acc = []

with Logger(write_to_file=True, file_name=file_name) as logger:
    logger.write("Starting run {} for {} epochs, and following params"
                    .format(run_id, num_epochs))
    logger.write("hash_dim: " + str(hash_dim))
    logger.write(optimizer_params)
    logger.write(custom_params)
    logger.write(batch_size)
    logger.write(loader_params)
    logger.write("====== START ======")

    for epoch in range(num_epochs):
        # ======================================================================
        # TRAINING
        # ======================================================================
        epoch_train_losses = []
        epoch_train_data_correct = []
        epoch_train_data_total = []
        epoch_train_bits_correct = []
        epoch_train_bits_total = []

        for num_iter, (X, y) in enumerate(loader_train):
            # set model to train mode
            model.train()
            # forward pass
            batch = X.to(device).float()
            label = y.to(device).float()
            outputs = model(batch)
            # calculating loss
            loss = F.binary_cross_entropy_with_logits(outputs, label)
            # loss to encourage code to be closer to -1 and 1
            loss += custom_params["beta"] * \
                        torch.abs((torch.abs(outputs)-1).sum()) / len(X)
            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate accuracy over training data
            label = label.byte()
            output_hash = (outputs.data > 0).to(device)
            # a classification is considered correct if the output matches the
            # target bits exactly
            num_correct = ((output_hash == label).sum(1) == hash_dim).sum() \
                                .item()
            # the number of bits that are correct
            num_bits_correct = (output_hash == label).sum().item()
            num_bits_total = reduce(lambda x, y: x*y, output_hash.shape)
            # calculate the percentage
            train_data_acc = num_correct * 100.0 / len(X)
            train_bits_acc = num_bits_correct * 100.0 / num_bits_total
            # save statistics
            epoch_train_losses.append(loss.item())
            epoch_train_data_correct.append(num_correct)
            epoch_train_data_total.append(len(X))
            epoch_train_bits_correct.append(num_bits_correct)
            epoch_train_bits_total.append(num_bits_total)

            if num_iter % 20 == 0:
                print(
                    "TRAIN epoch {} ".format(epoch) +
                    "iter {}: ".format(num_iter) +
                    "loss - {:.10f}, ".format(loss.item()) +
                    "correct/total - {}/{} ({:.2f}%), "
                        .format(num_correct, len(X), train_data_acc) +
                    "bits/total - {}/{} ({:.2f}%). "
                        .format(num_bits_correct, num_bits_total,
                                train_bits_acc)
                )

        # ======================================================================
        # VALIDATING
        # ======================================================================
        epoch_val_data_correct = []
        epoch_val_bits_correct = []
        epoch_val_data_total = []
        epoch_val_bits_total = []
        epoch_val_bmap = []
        best_map = float('-inf')

        for num_iter, (X, y) in enumerate(loader_val):
            model.eval()
            batch = X.float().to(device)
            label = y.float().to(device)
            outputs = model(batch)
            # calculate accuracy over validation data
            label = label.byte()
            output_hash = (outputs.data > 0).to(device)
            # number of data points correct
            num_correct = ((output_hash == label).sum(1) == hash_dim).sum()\
                                .item()
            # number of bits correct
            num_bits_correct = (output_hash == label).sum().item()
            num_bits_total = reduce(lambda x, y: x*y, output_hash.shape)
            # bits mean average precision
            bmap = calc_map(outputs, label).mean().item()
            if bmap > best_map:
                torch.save(model.state_dict(), checkpoint_path)
                best_map = bmap
            # append to history
            epoch_val_data_correct.append(num_correct)
            epoch_val_data_total.append(len(X))
            epoch_val_bits_correct.append(num_bits_correct)
            epoch_val_bits_total.append(num_bits_total)
            epoch_val_bmap.append(bmap)

            # calculate the loss and accuracy over the whole dataset by
            # averaging them
            final_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            final_train_data_correct = \
                sum(epoch_train_data_correct) / len(epoch_train_data_correct)
            final_train_data_total = \
                sum(epoch_train_data_total) / len(epoch_train_data_total)
            final_train_bits_correct = \
                sum(epoch_train_bits_correct) / len(epoch_train_bits_correct)
            final_train_bits_total = \
                sum(epoch_train_bits_total) / len(epoch_train_bits_total)

            final_val_data_correct = \
                sum(epoch_val_data_correct) / len(epoch_val_data_correct)
            final_val_data_total = \
                sum(epoch_val_data_total) / len(epoch_val_data_total)
            final_val_bits_correct = \
                sum(epoch_val_bits_correct) / len(epoch_val_bits_correct)
            final_val_bits_total = \
                sum(epoch_val_bits_total) / len(epoch_val_bits_total)
            final_val_bmap = \
                sum(epoch_val_bmap) / len(epoch_val_bmap)

        logger.write(
            "TRAIN epoch {} completed! Avg train loss - {:.6f}"
                .format(epoch, final_train_loss) +
            "\n\tTRAIN data correct/total: {}/{} ({:.2f}%), bits correct/total: {}/{} ({:.2f}%)"
                .format(round(final_train_data_correct),
                        round(final_train_data_total),
                        final_train_data_correct * 100.0 /
                        final_train_data_total,
                        round(final_train_bits_correct),
                        round(final_train_bits_total),
                        final_train_bits_correct * 100.0 /
                        final_train_bits_total) +
            "\n\tVAL   data correct/total: {}/{} ({:.2f}%), bits correct/total: {}/{} ({:.2f}%), map: {:.5f}"
                .format(round(final_val_data_correct),
                        round(final_val_data_total),
                        final_val_data_correct * 100.0 / final_val_data_total,
                        round(final_val_bits_correct),
                        round(final_val_bits_total),
                        final_val_bits_correct * 100.0 / final_val_bits_total,
                        final_val_bmap)
        )

    # ==========================================================================
    # TESTING
    # ==========================================================================
    best_model = DiscriminativeDeepHashing(hash_dim=hash_dim)
    best_model.load_state_dict(torch.load(checkpoint_path))

    epoch_test_data_correct = []
    epoch_test_bits_correct = []
    epoch_test_data_total = []
    epoch_test_bits_total = []
    epoch_test_bmap = []

    for num_iter, (X, y) in enumerate(loader_test):
        best_model.eval()
        batch = X.float().to(device)
        label = y.float().to(device)
        outputs = model(batch)
        # calculate accuracy over validation data
        label = label.byte()
        output_hash = (outputs.data > 0).to(device)
        # number of data points correct
        num_correct = ((output_hash == label).sum(1) == hash_dim).sum()\
                        .item()
        # number of bits correct
        num_bits_correct = (output_hash == label).sum().item()
        num_bits_total = reduce(lambda x, y: x*y, output_hash.shape)
        # bits mean average precision
        bmap = calc_map(outputs, label).mean().item()
        # append to history
        epoch_test_data_correct.append(num_correct)
        epoch_test_data_total.append(len(X))
        epoch_test_bits_correct.append(num_bits_correct)
        epoch_test_bits_total.append(num_bits_total)
        epoch_test_bmap.append(bmap)

    final_test_data_correct = \
        sum(epoch_test_data_correct) / len(epoch_test_data_correct)
    final_test_data_total = \
        sum(epoch_test_data_total) / len(epoch_test_data_total)
    final_test_bits_correct = \
        sum(epoch_test_bits_correct) / len(epoch_test_bits_correct)
    final_test_bits_total = \
        sum(epoch_test_bits_total) / len(epoch_test_bits_total)
    final_test_bmap = \
        sum(epoch_test_bmap) / len(epoch_test_bmap)

    logger.write(
        "Run completed! Test statistics: " +
        "\n\tTEST data correct/total: {}/{} ({:.2f}%), bits correct/total: {}/{} ({:.2f}%), map: {:.5f}"
            .format(round(final_test_data_correct),
                    round(final_test_data_total),
                    final_test_data_correct * 100.0 / final_test_data_total,
                    round(final_test_bits_correct),
                    round(final_test_bits_total),
                    final_test_bits_correct * 100.0 / final_test_bits_total,
                    final_test_bmap)
        )

    logger.write("====== END ======")
    logger.write("Completed run for {}".format(run_id))
