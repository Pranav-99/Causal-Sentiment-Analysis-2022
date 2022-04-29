import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
from collections import Counter
import tqdm
import random
import pickle
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score


class Model(nn.Module):
	def __init__(self, embed_dim=128, gru_out=32, z_dim=1):

		super(Model, self).__init__()
		self.embed_dim = embed_dim
		self.gru_out = gru_out
		self.z_dim = z_dim

		self.gru = nn.Sequential(
			nn.Embedding(2000, embed_dim),
			nn.GRU(embed_dim, gru_out, bidirectional=True, batch_first=True),
		)

		self.classifier = nn.Linear((2*gru_out)+z_dim, 1)

	def forward(self, x, z):
		x = self.gru(x)[1].transpose(0, 1).contiguous().view((x.shape[0], -1))
		assert(x.shape == (z.shape[0], self.gru_out*2))
		x = torch.cat((x, z), dim=1)
		x = self.classifier(x)
		return x

	def forward_all_z(self, x, all_z, p_z):
		batch_size = x.size(0)
		x = self.gru(x)[1].transpose(0, 1).contiguous().view((x.shape[0], -1))

		output = torch.zeros((batch_size, 1)).cuda()
		model_outputs = []
		for p_zi, z_val in zip(p_z, all_z):
			z_val = torch.unsqueeze(z_val, 0).repeat((batch_size, 1))
			x_z = torch.cat((x, z_val), dim=1)
			model_out_z = self.classifier(x_z)
			model_out_z = torch.sigmoid(model_out_z)
			output += p_zi.item() * model_out_z
			model_outputs.append(model_out_z)

		return output, model_outputs


class BasicModel(nn.Module):
	def __init__(self, embed_dim=128, gru_out=32):

		super(BasicModel, self).__init__()
		self.embed_dim = embed_dim
		self.gru_out = gru_out

		self.gru = nn.Sequential(
			nn.Embedding(2000, embed_dim),
			nn.GRU(embed_dim, gru_out, bidirectional=True, batch_first=True),
		)

		self.classifier = nn.Linear((2*gru_out), 1)

	def forward(self, x):
		x = self.gru(x)[1].transpose(0, 1).contiguous().view((x.shape[0], -1))
		x = self.classifier(x)
		return x


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
	    self.reset()

	def reset(self):
	    self.val = 0
	    self.avg = 0
	    self.sum = 0
	    self.count = 0

	def update(self, val, n=1):
	    self.val = val
	    self.sum += val * n
	    self.count += n
	    self.avg = self.sum / self.count


def train(X_train, y_train, z_train, model, optimizer, criterion, num_epochs=1, batch_size=256, lr=0.001):
      
	model.train()
	train_size = X_train.shape[0] 

	for epoch in range(num_epochs):

		train_loss_meter = AverageMeter()
		i = 0

		for batch in range(0, train_size, batch_size):

			start_index = batch
			end_index = min(batch + batch_size, train_size)

			batch_X = X_train[start_index:end_index].cuda()
			batch_z = z_train[start_index:end_index].cuda()

			batch_y = y_train[start_index:end_index].cuda()

			output = model(batch_X, batch_z)

			loss = criterion(output, batch_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss_meter.update(loss.item(), (end_index-start_index))

			if(i % 100 == 0):
				print("Epoch: {}, Iter: {}, Training Loss: {}".format(epoch, i, train_loss_meter.avg))

			i += 1


def predict(model, X_test, p_z, batch_size=256):

	model.eval()

	with torch.no_grad():

		if(len(p_z) == 4):
			all_z = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cuda()
		else:
			all_z = torch.tensor([[0], [1]], dtype=torch.float).cuda()

		test_size = X_test.shape[0]
		test_preds = torch.zeros((test_size, ))

		for batch in range(0, test_size, batch_size):

			start_index = batch
			end_index = min(batch + batch_size, test_size)

			batch_X = X_test[start_index:end_index].cuda()

			output, all_outputs = model.forward_all_z(batch_X, all_z, p_z)
   
			#print(torch.mean(all_outputs[0].squeeze(dim=1).cpu() - all_outputs[1].squeeze(dim=1).cpu()))

			test_preds[start_index:end_index] = output.squeeze(dim=1).cpu()

	model.train()

	return test_preds


def train_std(X_train, y_train, model, optimizer, criterion, num_epochs=1, batch_size=256, lr=0.001):
      
	model.train()
	train_size = X_train.shape[0] 

	for epoch in range(num_epochs):

		train_loss_meter = AverageMeter()
		i = 0

		for batch in range(0, train_size, batch_size):

			start_index = batch
			end_index = min(batch + batch_size, train_size)

			batch_X = X_train[start_index:end_index].cuda()

			batch_y = y_train[start_index:end_index].cuda()

			output = model(batch_X)

			loss = criterion(output, batch_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss_meter.update(loss.item(), (end_index-start_index))

			if(i % 100 == 0):
				print("Epoch: {}, Iter: {}, Training Loss: {}".format(epoch, i, train_loss_meter.avg))

			i += 1


def predict_std(model, X_test, batch_size=256):

	model.eval()

	with torch.no_grad():

		test_size = X_test.shape[0]
		test_preds = torch.zeros((test_size, ))

		for batch in range(0, test_size, batch_size):

			start_index = batch
			end_index = min(batch + batch_size, test_size)

			batch_X = X_test[start_index:end_index].cuda()

			output = torch.sigmoid(model.forward(batch_X))

			test_preds[start_index:end_index] = output.squeeze(dim=1).cpu()

	model.train()

	return test_preds


def prepare_data(all_train_df, test_df, ind=1):

	train_df = all_train_df[ind]
	train_df = train_df.sample(frac=1)

	train_df['text'] = train_df['text'].apply(lambda x: x.lower())
	test_df['text'] = test_df['text'].apply(lambda x: x.lower())

	train_df['text'] = train_df['text'].apply(lambda x: " ".join(wordpunct_tokenize(x)))
	test_df['text'] = test_df['text'].apply(lambda x: " ".join(wordpunct_tokenize(x)))

	tokenizer = Tokenizer(num_words=2000, lower=True, split=' ', filters='#%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
	tokenizer.fit_on_texts(train_df['text'].values)

	X_train = tokenizer.texts_to_sequences(train_df['text'].values)
	X_train = pad_sequences(X_train, maxlen=350)

	X_test = tokenizer.texts_to_sequences(test_df['text'].values)
	X_test = pad_sequences(X_test, maxlen=350)

	return X_train, X_test, train_df, test_df


if __name__ == "__main__":

	fp = open("train_dfs.pkl", "rb")
	all_train_df = pickle.load(fp)
	fp.close()

	fp = open("test_df.pkl", "rb")
	test_df = pickle.load(fp)
	fp.close()

	confounders = ['user_pop', 'take_out']

	for dataset_index in range(1, 10):

		random.seed(0)
		torch.manual_seed(0)
		np.random.seed(0)

		print("\n-------------\nDataset Bias {}\n\n".format(dataset_index))

		X_train, X_test, train_df, test_df = prepare_data(all_train_df, test_df, dataset_index)

		X_train_tensor = torch.from_numpy(X_train)
		y_train_tensor = torch.from_numpy(train_df['label'].to_numpy()).unsqueeze(dim=1).float()
		z_train_tensor = torch.from_numpy(train_df[confounders].to_numpy())

		X_test_tensor = torch.from_numpy(X_test)
		y_test_tensor = torch.from_numpy(test_df['label'].to_numpy()).unsqueeze(dim=1).float()
		z_test_tensor = torch.from_numpy(test_df[confounders].to_numpy())

		num_epochs = 1
		batch_size = 256
		lr = 0.001

		model = Model(z_dim=len(confounders)).cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		criterion = nn.BCEWithLogitsLoss()

		train(X_train_tensor, y_train_tensor, z_train_tensor, model, optimizer, criterion, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

		p_z = Counter(map(lambda x: tuple(x), train_df[confounders].values))
		p_z = np.array([p_z[x] for x in sorted(p_z.keys())], dtype=float)
		p_z /= p_z.sum()

		pred_test = predict(model, X_test_tensor, p_z, batch_size=batch_size)
		y_pred = np.round(pred_test.numpy())

		pred_train = predict(model, X_train_tensor, p_z, batch_size=batch_size)
		y_pred_train = np.round(pred_train.numpy())

		print("\nDataset Bias {}\n".format(dataset_index))		
		print("Test Accuracy: {}".format(accuracy_score(test_df['label'], y_pred)))
		print("Train Accuracy: {}".format(accuracy_score(train_df['label'], y_pred_train)))

		model_std = BasicModel().cuda()
		optimizer_std = torch.optim.Adam(model_std.parameters(), lr=lr)

		train_std(X_train_tensor, y_train_tensor, model_std, optimizer_std, criterion, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

		pred_test_std = predict_std(model_std, X_test_tensor, batch_size=batch_size)
		y_pred_std = np.round(pred_test_std.numpy())

		pred_train_std = predict_std(model_std, X_train_tensor, batch_size=batch_size)
		y_pred_train_std = np.round(pred_train_std.numpy())

		print("Test Accuracy (std): {}".format(accuracy_score(test_df['label'], y_pred_std)))
		print("Train Accuracy (std): {}".format(accuracy_score(train_df['label'], y_pred_train_std)))

		print("\n\n-------------\n\n", dataset_index)