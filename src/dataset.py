import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class RanzcrDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_folder = image_folder
        self.from_image_folder = from_image_folder
        self.transform = transform

        self.mode = mode
        self.clahe = clahe
        self.mix = mix
        if self.clahe or self.mix:
            self.clahe_transform = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16, 16))

        self.cols = [
            'ETT - Abnormal', 'ETT - Borderline',
            'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
            'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
            'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
                     ]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.from_image_folder:
            img_path = os.path.join(self.image_folder, row["StudyInstanceUID"] + ".jpg")
        else:
            img_path = row.img_path
        images = cv2.imread(img_path).astype(np.float32)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if self.clahe:
            single_channel = images[:, :, 0].astype(np.uint8)
            single_channel = self.clahe_transform.apply(single_channel)
            images = np.array([
                single_channel,
                single_channel,
                single_channel
            ]).transpose(1, 2, 0)
        elif self.mix:
            single_channel = images[:, :, 0].astype(np.uint8)
            clahe_channel = self.clahe_transform.apply(single_channel)
            hist_channel = cv2.equalizeHist(single_channel)
            images = np.array([
                single_channel,
                clahe_channel,
                hist_channel
            ]).transpose(1, 2, 0)

        if self.transform is not None:
            images = self.transform(image=images.astype(np.uint8))['image']
        else:
            images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        if self.mode == "train":
            label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": torch.tensor(label, dtype=torch.float)
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }

