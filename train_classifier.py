import sys

import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm

from train_vlisulization_visdom import Visualizer
from dataloader import CelebA
from classifier import Classifier


def train(model: Classifier, dataset, viz: Visualizer, save_name, num_epoch,lr):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
    bce = nn.BCEWithLogitsLoss()
    # mse = nn.MSELoss()
    model.train()

    for epoch in range(num_epoch):
        losses = []
        kld_losses = []
        mse_losses = []
        for i, (x, label, _) in tqdm(enumerate(dataloader)):
            label = label.cuda()
            label = label.unsqueeze(1).float()
            x = x.cuda()
            model.zero_grad()
            y, mu, log_var = model(x)

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
            bce_loss = bce(y, label) #change mse to bce loss

            kld_weight = 5e-6
            loss = bce_loss + kld_weight * kld_loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            #to constrain the kld_loss to be small
            kld_losses.append(kld_loss.item())
            mse_losses.append(bce_loss.item())

            if viz.every_n_step(num_epoch):
                # For visdom, we need to use the mean of the losses
                loss_mean = np.mean(losses)
                kld_loss_mean = np.mean(kld_losses)
                mse_loss_mean = np.mean(mse_losses)
                print(f"loss: {loss_mean} kld_loss: {kld_loss_mean} bce_loss: {mse_loss_mean}")
                viz.plot("loss", loss_mean)
                viz.plot("kld_loss", kld_loss_mean)
                viz.plot("bce_loss", mse_loss_mean)
                losses = []
                kld_losses = []
                mse_losses = []

            if viz.every_n_step(num_epoch):
                for i in range(8):
                    viz.imShow(f"l:{label[i].item()}, y:{round(y[i].item(), 2)}", x[i])
            viz.tic()
        print("save pickle in training!")
        torch.save(model.state_dict(),
                   "pickles" + os.sep + f"{save_name}_epoch_{epoch}_{round(loss_mean, 7)}" + ".pickle")


# if __name__ == '__main__':
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if(len(sys.argv)>2):
#         retrain_load = sys.argv[1]
#     else:
#         retrain_load = "None"
#     classifier = Classifier().to(DEVICE)
#
#     #if need retrain, uncomment following line
#     load_pickle = retrain_load
#     if os.path.exists("pickles/" + load_pickle):
#         print("load pickle!")
#         with open("pickles/" + load_pickle, "rb") as f:
#             state_dict = torch.load(f, map_location=DEVICE)
#         classifier.load_state_dict(state_dict)
#         print("load pickle success")
#     # TODO change of save_name
#     save_name = "classifier_100_leakrelu_sgd"
#     viz = Visualizer(save_name, x_axis="step")
#     dataset = CelebA("/home/robotlab/Desktop/img_align_celeba_png", usage="train")
#     train(classifier, dataset, viz=viz, save_name=save_name, num_epoch=100, lr=1e-4)
    #cross validation on training set, we need to split ourself and retain the best model
    # dataset = CelebA("/home/robotlab/Desktop/img_align_celeba_png", usage="test")
    # train(classifier, dataset, viz=viz, save_name=save_name)
    # dataset = CelebA("/home/robotlab/Desktop/img_align_celeba_png", usage="valid")
    # train(classifier, dataset, viz=viz, save_name=save_name)