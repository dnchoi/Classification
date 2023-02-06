import torch
from PIL import Image


class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, classes, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])

        if self.transform is not None:
            img = self.transform(img)

        label = 0
        for i in range(len(self.classes)):
            if self.classes[i] in self.file_list:
                label = i
            else:
                pass

        return img, label
