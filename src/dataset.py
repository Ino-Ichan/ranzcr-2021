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


####################
# Dataset for segmentation
####################

class RanzcrSegDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder=None,
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

        mask = cv2.imread('/data/additional/train_lung_masks/train_lung_masks/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
        mask = (mask > 127) * 1

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
            transformed = self.transform(image=images.astype(np.uint8), mask=mask.astype(np.uint8))
            images = transformed['image']
            mask = transformed['mask']
        else:
            images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        if self.mode == "train":
            #             label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": mask
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }

####################
# Dataset for cvc
####################

class RanzcrCVCDataset0(Dataset):
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
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
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


import albumentations
from albumentations.pytorch import ToTensorV2

class RanzcrCVCDataset1(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 padding=25,
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
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                     ]

        self.shared_transform = albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1)
            ])
        self.padding = padding

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

        mask = cv2.imread('/data/additional/train_lung_masks/train_lung_masks/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
        mask = (mask > 127) * 1

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
            transformed = self.transform(image=images.astype(np.uint8), mask=mask.astype(np.uint8))
            images = transformed['image']
            mask = transformed['mask']
        # else:
        #     images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # lung crop
        x0x1 = np.where(mask.max(0) == 1)[0]
        y0y1 = np.where(mask.max(1) == 1)[0]
        x0 = np.max([x0x1[0] - self.padding, 0])
        x1 = np.min([x0x1[-1] + self.padding, mask.shape[1]])
        y0 = np.max([y0y1[0] - self.padding, 0])
        y1 = np.min([y0y1[-1] + self.padding, mask.shape[0]])

        images = images[y0:y1, x0:x1]
        # mask = mask[y0:y1, x0:x1]

        images = self.shared_transform(image=images)['image']

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


class RanzcrCVCDataset4(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 padding=25,
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

        self.shared_transform = albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1)
            ])
        self.padding = padding

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

        mask = cv2.imread('/data/additional/train_lung_masks/train_lung_masks/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
        mask = (mask > 127) * 1

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
            transformed = self.transform(image=images.astype(np.uint8), mask=mask.astype(np.uint8))
            images = transformed['image']
            mask = transformed['mask']
        # else:
        #     images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # lung crop
        x0x1 = np.where(mask.max(0) == 1)[0]
        y0y1 = np.where(mask.max(1) == 1)[0]
        x0 = np.max([x0x1[0] - self.padding, 0])
        x1 = np.min([x0x1[-1] + self.padding, mask.shape[1]])
        y0 = np.max([y0y1[0] - self.padding, 0])
        y1 = np.min([y0y1[-1] + self.padding, mask.shape[0]])

        images = images[y0:y1, x0:x1]
        # mask = mask[y0:y1, x0:x1]

        images = self.shared_transform(image=images)['image']

        if self.mode == "train":
            label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": torch.tensor(label, dtype=torch.float),
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }

##################################################################################
### With segmentation
##################################################################################


class RanzcrCVCDataset5(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 padding=25,
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
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                     ]

        self.shared_transform = albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1)
            ])
        self.padding = padding

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

        target_mask = cv2.imread('/data/additional/cvc_line_seg/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.resize(target_mask, (images.shape[1], images.shape[0]))
        target_mask = (target_mask > 127) * 1

        mask = cv2.imread('/data/additional/train_lung_masks/train_lung_masks/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
        mask = (mask > 127) * 1

        mask = np.array([
            mask,
            target_mask
        ]).transpose(1, 2, 0)

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
            transformed = self.transform(image=images.astype(np.uint8), mask=mask.astype(np.uint8))
            images = transformed['image']
            mask = transformed['mask']
        # else:
        #     images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # lung crop
        x0x1 = np.where(mask[:, :, 0].max(0) == 1)[0]
        y0y1 = np.where(mask[:, :, 0].max(1) == 1)[0]
        x0 = np.max([x0x1[0] - self.padding, 0])
        x1 = np.min([x0x1[-1] + self.padding, mask.shape[1]])
        y0 = np.max([y0y1[0] - self.padding, 0])
        y1 = np.min([y0y1[-1] + self.padding, mask.shape[0]])

        images = images[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]

        transformed = self.shared_transform(image=images, mask=mask)
        images = transformed['image']
        mask = transformed['mask']

        if self.mode == "train":
            label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": torch.tensor(label, dtype=torch.float),
                "mask": mask[:, :, 1]
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }


class RanzcrCVCDataset5Eval(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 clahe=False,
                 mix=False,
                 padding=25,
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
            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        ]

        self.shared_transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1)
        ])
        self.padding = padding

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

        # target_mask = cv2.imread('/data/additional/cvc_line_seg/' + row.StudyInstanceUID + '.jpg',
        #                          cv2.IMREAD_GRAYSCALE)
        # target_mask = cv2.resize(target_mask, (images.shape[1], images.shape[0]))
        # target_mask = (target_mask > 127) * 1

        mask = cv2.imread('/data/additional/train_lung_masks/train_lung_masks/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (images.shape[1], images.shape[0]))
        mask = (mask > 127) * 1

        # mask = np.array([
        #     mask,
        #     target_mask
        # ]).transpose(1, 2, 0)

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
            transformed = self.transform(image=images.astype(np.uint8), mask=mask.astype(np.uint8))
            images = transformed['image']
            mask = transformed['mask']
        # else:
        #     images = images.transpose(2, 0, 1)

        # normalize image
        # images = images / 255
        # image net normalize
        # images = (images - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # lung crop
        x0x1 = np.where(mask.max(0) == 1)[0]
        y0y1 = np.where(mask.max(1) == 1)[0]
        x0 = np.max([x0x1[0] - self.padding, 0])
        x1 = np.min([x0x1[-1] + self.padding, mask.shape[1]])
        y0 = np.max([y0y1[0] - self.padding, 0])
        y1 = np.min([y0y1[-1] + self.padding, mask.shape[0]])

        images = images[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]

        transformed = self.shared_transform(image=images, mask=mask)
        images = transformed['image']
        mask = transformed['mask']

        if self.mode == "train":
            label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": torch.tensor(label, dtype=torch.float),
                "mask": mask
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }