from models.SCMModel import StructuralCausalModel
from models.GAN import GANmodel

from utils import *
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def train_scm(X_train, y_train, c_train, model, optimizer, num_epochs=1, batch_size=256, lr=0.001):

    model.train()
    train_size = X_train.shape[0]

    train_loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    y_xz_loss_meter = AverageMeter()
    c_z_loss_meter = AverageMeter()
    KL_loss_meter = AverageMeter()

    tb_writer = SummaryWriter(f'/content/runs/{datetime.now()}'.replace(' ', '_'))
    tb_writer.add_text(f"hyperparams",
                       "batch_size: {batch_size}, num_epochs: {num_epochs}, lr: {lr},"
                       " model lambdas: {model.lambda_recon, model.lambda_y_xz, model.lambda_c_z, model.lambda_KL}")
    for epoch in range(num_epochs):

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

            tb_writer.add_scalar('train/loss', loss.item(), start_index)
            tb_writer.add_scalar('train/recon_loss', L_recon.item(), start_index)
            tb_writer.add_scalar('train/y_xz_loss', L_y_xz.item(), start_index)
            tb_writer.add_scalar('train/c_z_loss', L_c_z.item(), start_index)
            tb_writer.add_scalar('train/KL_loss', L_KL.item(), start_index)

            if(i % 100 == 0):

                print("Epoch: {}, Iter: {}, Training Loss: {}".format(epoch, i, train_loss_meter.avg), end='')
                # print(f"loss, L_recon, L_y_xz, L_c_z, L_KL: {loss.item(), L_recon.item(), L_y_xz.item(), L_c_z.item(), L_KL.item()}")
                print(", recon loss: {}".format(recon_loss_meter.avg), end='')
                print(", y_xz loss: {}".format(y_xz_loss_meter.avg), end='')
                print(", c_z loss: {}".format(c_z_loss_meter.avg), end='')
                print(", KL loss: {}".format(KL_loss_meter.avg))

            i += 1


def generate_z_dataset(X_train, y_train, c_train, model, batch_size=1024):
    
    model.eval()
    train_size = X_train.shape[0]
    z_values = []

    with torch.no_grad():
      for batch in range(0, train_size, batch_size):
          start_index = batch
          end_index = min(batch + batch_size, train_size)

          batch_X = X_train[start_index:end_index].cuda()
          batch_c = c_train[start_index:end_index].cuda()

          batch_y = y_train[start_index:end_index].cuda()

          batch_z = model.generate_z(batch_X, batch_y, batch_c).detach().cpu().numpy()
          z_values.append(batch_z)

    z_values = np.vstack(z_values)
    return z_values


def predict_dataset(model: StructuralCausalModel, X_test, gan: GANmodel, batch_size=256):
    
    model.eval()

    with torch.no_grad():

        test_size = X_test.shape[0]
        test_preds = torch.zeros((test_size, ))

        for batch in range(0, test_size, batch_size):

            start_index = batch
            end_index = min(batch + batch_size, test_size)

            batch_X = X_test[start_index:end_index].cuda()

            output = model.predict(batch_X, gan)

            test_preds[start_index:end_index] = output.squeeze(dim=1).cpu()

    model.train()

    return test_preds


def main():

    fp = open("train_dfs.pkl", "rb")
    all_train_df = pickle.load(fp)
    fp.close()

    fp = open("test_df.pkl", "rb")
    test_df = pickle.load(fp)
    fp.close()

    confounder = ['user_pop', 'take_out']

    # dataset_index = 1
    for dataset_index in range(1, 10):
    # if True:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        print("\n-------------\nDataset Bias {}\n\n".format(dataset_index))
        X_train, X_test, train_df, test_df = prepare_data(all_train_df, test_df, dataset_index)

        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(train_df['label'].to_numpy()).unsqueeze(dim=1).float()
        c_train_tensor = torch.from_numpy(train_df[confounder].to_numpy()) #.unsqueeze(dim=1)

        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(test_df['label'].to_numpy()).unsqueeze(dim=1).float()
        c_test_tensor = torch.from_numpy(test_df[confounder].to_numpy()) #.unsqueeze(dim=1)

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        num_epochs = 10
        batch_size = 128
        lr = 0.001

        model = StructuralCausalModel(lambda_KL=0.5, c_dim=len(confounder)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_scm(
            X_train_tensor, y_train_tensor, c_train_tensor.float(), model, optimizer,
            num_epochs=num_epochs, batch_size=batch_size, lr=lr,
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        z_dataset_train = generate_z_dataset(X_train_tensor, y_train_tensor, c_train_tensor.float(), model)
        z_dataset_test = generate_z_dataset(X_test_tensor, y_test_tensor, c_test_tensor.float(), model)

        gan_config = {}
        gan_config['device'] = device
        gan_config['g_input_dim'] = 16
        gan_config['g_hidden_dim'] = 8
        gan_config['g_output_dim'] = 16
        gan_config['d_hidden_dim'] = 4
        gan_config['d_output_dim'] = 1
        gan_config['g_lr'] = 0.001
        gan_config['d_lr'] = 0.001
        gan_config['epochs'] = 1
        gan_config['gamma'] = 1.0
        gan_config['mode'] = 'LSGAN'
        gan_config['disc_steps'] = 5
        gan_config['g_nhidden'] = 2

        gan_config['batch_size'] = 256

        gan = GANmodel(gan_config)
        z_dataset_train = torch.from_numpy(z_dataset_train)
        z_dataset_test = torch.from_numpy(z_dataset_test)
        gan.train(z_dataset_train, z_dataset_test)

        pred_test = predict_dataset(model, X_test_tensor, gan)

        y_pred = np.round(pred_test.cpu().numpy())

        print("\nDataset Bias {}\n".format(dataset_index))      
        print("Accuracy: {}".format(accuracy_score(test_df['label'], y_pred)))

        torch.save(model.state_dict(), f"scm_dataset_index={dataset_index}.pt")
        torch.save(gan.g.state_dict(), f"gan_generator_dataset_index={dataset_index}.pt")
        torch.save(gan.d.state_dict(), f"gan_discriminator_dataset_index={dataset_index}.pt")

        print("\n\n-------------\n\n", dataset_index)


if __name__ == '__main__':
    main()
