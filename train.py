from curses import termattrs
from unittest import TextTestResult
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import shutil
import zipfile
import os
import time

from glob import glob
from tqdm import tqdm
import logging
from torchsummary import summary

#INIT Custom librarys
from lib.dataloader import dog_cat_dataloader

logging.basicConfig(filename='train.log',level=logging.INFO)

def fit(model_name, model, criterion, optimizer, epochs, train_loader, valid_loader):
	train_loss = 0
	train_acc = 0
	train_correct = 0
	train_losses = []
	train_accuracies = []
	valid_losses = []
	valid_accuracies = []
	if not os.path.exists(model_name):
		os.mkdir(model_name)
	for epoch in tqdm(range(epochs)):
		start = time.time()
		for train_x, train_y in tqdm(train_loader):
			model.train()
			train_x, train_y = train_x.to(device), train_y.to(device).float()
			optimizer.zero_grad()
			pred = model(train_x)
			loss = criterion(pred, train_y)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			y_pred = pred.cpu()
			y_pred[y_pred >= 0.5] = 1
			y_pred[y_pred < 0.5] = 0
			train_correct += y_pred.eq(train_y.cpu()).int().sum()
		# validation data check
		valid_loss = 0
		valid_acc = 0
		valid_correct = 0
		for valid_x, valid_y in tqdm(valid_loader):
			with torch.no_grad():
				model.eval()
				valid_x, valid_y = valid_x.to(device), valid_y.to(device).float()
				pred = model(valid_x)
				loss = criterion(pred, valid_y)
				valid_loss += loss.item()
				y_pred = pred.cpu()
				y_pred[y_pred >= 0.5] = 1
				y_pred[y_pred < 0.5] = 0
				valid_correct += y_pred.eq(valid_y.cpu()).int().sum()
				train_acc = train_correct/len(train_loader.dataset)
				valid_acc = valid_correct/len(valid_loader.dataset)
				print(f'{time.time() - start:.3f}sec : [Epoch {epoch+1}/{epochs}] -> train loss: {train_loss/len(train_loader):.4f}, train acc: {train_acc*100:.3f}% / valid loss: {valid_loss/len(valid_loader):.4f}, valid acc: {valid_acc*100:.3f}%')
				train_losses.append(train_loss/len(train_loader))
				train_accuracies.append(train_acc)
				valid_losses.append(valid_loss/len(valid_loader))
				valid_accuracies.append(valid_acc)
				train_loss = 0
				train_acc = 0
				train_correct = 0
		plt.plot(train_losses, label='loss')
		plt.plot(train_accuracies, label='accuracy')
		plt.legend()
		plt.title('train loss and accuracy')
		# plt.show()
		plt.savefig(os.path.join(model_name, model_name+"_train_loss_accuracy_"+str(epoch)+".png"))
		plt.cla()
  
		plt.plot(valid_losses, label='loss')
		plt.plot(valid_accuracies, label='accuracy')
		plt.legend()
		plt.title('valid loss and accuracy')
		# plt.show()
		plt.savefig(os.path.join(model_name, model_name+"_valid_loss_accuracy_"+str(epoch)+".png"))
		plt.cla()

	torch.save(model, os.path.join(model_name, model_name+".pt"))

dataset_root = './datasets/'
train_data = os.path.join(dataset_root, "train")
test_data = os.path.join(dataset_root, "test")

dog_file = glob(os.path.join(train_data, "dog.*.jpg"))
cat_file = glob(os.path.join(train_data, "cat.*.jpg"))

_train_data = os.path.join(dataset_root, "train_data")
_vaild_data = os.path.join(dataset_root, "vaild_data")

if not os.path.exists(_train_data):
	os.mkdir(_train_data)
if not os.path.exists(_vaild_data):
	os.mkdir(_vaild_data)

n = 0
for i, j in tqdm(zip(dog_file, cat_file)):
	if n < 0.95*len(dog_file):
		shutil.copy(i, _train_data)
		shutil.copy(j, _train_data)
	else:
		shutil.copy(i, _vaild_data)
		shutil.copy(j, _vaild_data)
	
	n+=1

train_transform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((256, 256)),
	torchvision.transforms.RandomCrop(224),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.ToTensor(),
])

test_transform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((224,244)),
	torchvision.transforms.ToTensor(),
])


train_dog_dataset = dog_cat_dataloader(dog_file[:10000], _train_data, transform=train_transform)
train_cat_dataset = dog_cat_dataloader(cat_file[:10000], _train_data, transform=train_transform)
valid_dog_dataset = dog_cat_dataloader(dog_file[10000:11250], _vaild_data, transform=test_transform)
valid_cat_dataset = dog_cat_dataloader(cat_file[10000:11250], _vaild_data, transform=test_transform)
test_dog_dataset = dog_cat_dataloader(dog_file[11250:], test_data, transform=test_transform)
test_cat_dataset = dog_cat_dataloader(cat_file[11250:], test_data, transform=test_transform)

train_dataset = torch.utils.data.ConcatDataset([train_dog_dataset, train_cat_dataset])
valid_dataset = torch.utils.data.ConcatDataset([valid_dog_dataset, valid_cat_dataset])
test_dataset = torch.utils.data.ConcatDataset([test_dog_dataset, test_cat_dataset])

logging.info("train dataset : {}".format(len(train_dataset)))
logging.info("valid dataset : {}".format(len(valid_dataset)))
logging.info("test dataset : {}".format(len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32+32+32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32+32+32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32+32+32, shuffle=True)


samples, labels = iter(train_loader).next()
classes = {0:'cat', 1:'dog'}
# fig = plt.figure(figsize=(16,24))
# for i in range(24):
# 	a = fig.add_subplot(4,6,i+1)
# 	a.set_title(classes[labels[i].item()])
# 	a.axis('off')
# 	a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
# plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
# plt.cla()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_list = [{"pre_resnet18":torchvision.models.resnet18(pretrained=True)},{"pre_resnet34":torchvision.models.resnet34(pretrained=True)}, \
    		  {"pre_resnet50":torchvision.models.resnet50(pretrained=True)},{"pre_resnet101":torchvision.models.resnet101(pretrained=True)},{"pre_resnet152":torchvision.models.resnet152(pretrained=True)}, \
              {"resnet18":torchvision.models.resnet18(pretrained=False)},{"resnet34":torchvision.models.resnet34(pretrained=False)}, \
    		  {"resnet50":torchvision.models.resnet50(pretrained=False)},{"resnet101":torchvision.models.resnet101(pretrained=False)},{"resnet152":torchvision.models.resnet152(pretrained=False)}]

model_names = ["pre_resnet18","pre_resnet34","pre_resnet50","pre_resnet101","pre_resnet152", \
    		   "resnet18","resnet34","resnet50","resnet101","resnet152"]
for model_name, init_model in zip(model_names, model_list):
	logging.info(model_name)
	model = init_model[model_name]
	num_ftrs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Dropout(0.5),
		nn.Linear(num_ftrs, 1024),
		nn.Dropout(0.2),
		nn.Linear(1024, 512),
		nn.Dropout(0.1),
		nn.Linear(512, 1),
		nn.Sigmoid()
	)

#	model.cuda()

	input_dim = samples[0].numpy().shape
	logging.info(init_model[model_name])

	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

	fit(model_name, model, criterion, optimizer, 100, train_loader, valid_loader)
	
