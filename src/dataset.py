import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 transform=None,
                 mode="train"
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_folder = image_folder
        self.transform = transform

        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row.image_id

        img_path = os.path.join(self.image_folder, image_name)
        images = cv2.imread(img_path).astype(np.float32)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        # if self.image_size != "original":
        #     images = cv2.resize(images, (self.image_size, self.image_size))

        if self.transform is not None:
            images = self.transform(image=images.astype(np.uint8))['image']
        else:
            images = images.transpose(2, 0, 1)

        # normalize image
        images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        if self.mode == "train":
            label = row.label
            return {
                "image": torch.tensor(images, dtype=torch.float),
                "target": torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }

