import torch
import os
import PIL
import numpy as np

class dog_cat_dataloader(torch.utils.data.Dataset):
	def __init__(self, files, mode='train', transform=None):
		self.files = files
		self.mode = mode
		self.transform = transform
		
		if 'cat' in files[0]:
			self.label = 0
		else:
			self.label = 1
	
	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		img = PIL.Image.open(self.files[index])
		if self.transform:
			img = self.transform(img)
		if 'test' in self.mode:
			return img, self.files[index]
		else:
			return img, np.array([self.label])