from models import *
from utils import *
import torch
from torch import nn
import torch.nn.functional as F

def train_scm(X_train, y_train, c_train, model, optimizer, num_epochs=1, batch_size=256, lr=0.001):

    model.train()
    train_size = X_train.shape[0]

    for epoch in range(num_epochs):

        train_loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        y_xz_loss_meter = AverageMeter()
        c_z_loss_meter = AverageMeter()
        KL_loss_meter = AverageMeter()

        i = 0

        for batch in range(0, train_size, batch_size):

            start_index = batch
            end_index = min(batch + batch_size, train_size)

            batch_X = X_train[start_index:end_index].cuda()
            batch_c = c_train[start_index:end_index].cuda()

            batch_y = y_train[start_index:end_index].cuda()

            loss, L_recon, L_y_xz, L_c_z, L_KL = model(batch_X, batch_y, batch_c, return_all_losses=True)

            # loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), (end_index-start_index))
            recon_loss_meter.update(L_recon.item(), (end_index-start_index))
            y_xz_loss_meter.update(L_y_xz.item(), (end_index-start_index))
            c_z_loss_meter.update(L_c_z.item(), (end_index-start_index))
            KL_loss_meter.update(L_KL.item(), (end_index-start_index))

            if(i % 100 == 0):

                print("Epoch: {}, Iter: {}, Training Loss: {}".format(epoch, i, train_loss_meter.avg))
                print(f"loss, L_recon, L_y_xz, L_c_z, L_KL: {loss.item(), L_recon.item(), L_y_xz.item(), L_c_z.item(), L_KL.item()}")
                # print(", recon loss: {}".format(epoch, i, recon_loss_meter.avg), end='')
                # print(", y_xz loss: {}".format(epoch, i, y_xz_loss_meter.avg), end='')
                # print(", c_z loss: {}".format(epoch, i, c_z_loss_meter.avg), end='')
                # print(", KL loss: {}".format(epoch, i, KL_loss_meter.avg))

            i += 1