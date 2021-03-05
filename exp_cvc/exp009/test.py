import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, roc_auc_score

from tqdm import tqdm
import copy
import argparse
import os, sys, yaml

sys.path.append('./')
from src.logger import setup_logger, LOGGER
from src.models import Net, RANZCRResNet200D
from src.losses import LabelSmoothingCrossEntropy
from src.dataset import RanzcrCVCDataset5 as RanzcrDataset #, RanzcrCVCDataset5Eval
# from src.dataset import RanzcrDataset
from src.utils import plot_sample_images
from src.augmix import RandomAugMix
from src.seg_loss.lovasz_losses import SymmetricLovaszLoss

import segmentation_models_pytorch as smp

import warnings

import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class RanzcrCVCDataset5(torch.utils.data.Dataset):
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

        target_mask = cv2.imread('/data/additional/cvc_line_seg/' + row.StudyInstanceUID + '.jpg',
                          cv2.IMREAD_GRAYSCALE)
        if target_mask is None:
            target_mask = mask
        target_mask = cv2.resize(target_mask, (images.shape[1], images.shape[0]))
        target_mask = (target_mask > 127) * 1

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
                "mask": mask[:, :, 1],
                "StudyInstanceUID": row.StudyInstanceUID
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }


def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def ousm_loss(error, k=2):
    # ousm, drop large k sample
    bs = error.shape[0]
    if len(error.shape) == 2:
        error = error.mean(1)
    _, idxs = error.topk(bs - k, largest=False)
    error = error.index_select(0, idxs)
    return error


# Freeze batchnorm 2d
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


# change targets, https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/186492
def forward(data, model, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    targets = data["target"].to(device)
    mask = data["mask"].to(device)
    pred_mask, pred = model(inputs)
    pred_labels = pred.sigmoid()

    loss_label = criterion[0](pred, targets).mean()
    loss_mask = criterion[1](pred_mask, mask).mean()
    loss = loss_label + loss_mask
    # loss = ousm_loss(loss, 3).mean()

    return loss, pred.detach().cpu().numpy().tolist(),\
           targets.cpu().numpy().tolist(), pred_labels.detach().cpu().numpy().tolist(),\
           loss_label, loss_mask, pred_mask.sigmoid().detach().cpu().numpy().tolist()

def forward2(data, model, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    targets = data["target"].to(device)
    mask = data["mask"].to(device)
    pred_mask = model(inputs)
    # pred_labels = pred.sigmoid()

    # loss_label = criterion[0](pred, targets).mean()
    loss_mask = criterion[1](pred_mask, mask).mean()
    loss = loss_mask
    # loss = ousm_loss(loss, 3).mean()

    return loss, None,\
           None, None,\
           None, loss_mask, pred_mask.sigmoid().detach().cpu().numpy().tolist(), data["StudyInstanceUID"]


def get_train_transforms(image_size):
    return albumentations.Compose([
           albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0., rotate_limit=30, p=0.8),
           albumentations.RandomResizedCrop(image_size, image_size, scale=(0.7, 1), p=0.5),
           albumentations.HorizontalFlip(p=0.5),
           albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
           albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
           albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
           # albumentations.OneOf([
           #     albumentations.OpticalDistortion(distort_limit=1.0),
           #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
           #     albumentations.ElasticTransform(alpha=3),
           # ], p=0.2),
           albumentations.OneOf([
               albumentations.GaussNoise(var_limit=[10, 50]),
               albumentations.GaussianBlur(),
               albumentations.MotionBlur(),
               albumentations.MedianBlur(),
           ], p=0.2),
          # albumentations.Resize(image_size, image_size),
          # albumentations.OneOf([
          #     albumentations.augmentations.transforms.JpegCompression(),
          #     albumentations.augmentations.transforms.Downscale(scale_min=0.1, scale_max=0.15),
          # ], p=0.2),
          # albumentations.imgaug.transforms.IAAPiecewiseAffine(p=0.2),
          # albumentations.imgaug.transforms.IAASharpen(p=0.2),
          albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
          # albumentations.Normalize(
          #     mean=[0.485, 0.456, 0.406],
          #     std=[0.229, 0.224, 0.225],
          # ),
          # ToTensorV2(p=1)
])



def get_val_transforms(image_size):
    return None



if __name__ == "__main__":
    print('Start!!!')
    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('-y', '--yaml_path', type=str,
                        help='configを書いたyamlのPath。例）-y ../config/exp0001.yaml')

    args = parser.parse_args()

    yaml_path = args.yaml_path
    yaml_path = args.yaml_path
    if os.path.isfile(yaml_path):
        with open(yaml_path) as file:
            cfg = yaml.safe_load(file.read())
    else:
        print('Error: No such yaml file')
        sys.exit()
    # seed_everything
    seed_torch()

    # output
    exp_name = cfg["exp_name"]  # os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join("/workspace/output_cvc", exp_name)
    # path
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    oof_path = output_path + "/oof"
    sample_img_path = output_path + "/sample_img"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)

    save_mask = output_path + "/save_mask"
    os.makedirs(save_mask, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log.txt")
    setup_logger(out_file=log_path)
    LOGGER.info("config")
    LOGGER.info(cfg)
    LOGGER.info('')

    debug = cfg["debug"]
    if debug:
        LOGGER.info("Debug!!!!!")

    # params
    device_id = cfg["device_id"]
    try:
        device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
    except Exception as e:
        LOGGER.info('GPU is not available, {}'.format(e))
        sys.exit()

    print(device)

    #######################################
    ## params
    #######################################
    model_name = cfg["model_name"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    n_workers = cfg["n_workers"]
    n_epochs = cfg["n_epochs"]
    start_epoch = cfg["start_epoch"]
    transform = cfg["transform"]
    hold_out = cfg["hold_out"]
    accumulation_steps = cfg["accumulation_steps"]
    early_stopping_steps = cfg["early_stopping_steps"]
    freeze_bn = cfg["freeze_bn"]
    clahe = cfg["clahe"]
    mix = cfg["mix"]
    img_dir = cfg["img_dir"]

    # cols = [
    #     'ETT - Abnormal', 'ETT - Borderline',
    #     'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
    #     'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
    #     'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
    # ]
    cols = [
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
    ]

    df_anot = pd.read_csv('/data/train_annotations.csv')
    anot_index = df_anot[df_anot.label.str.contains('CVC')].StudyInstanceUID.unique()


    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg["df_train_path"])
    # df = df[df.StudyInstanceUID.isin(anot_index)].reset_index()

    cv_list = hold_out if hold_out else [0, 1, 2, 3, 4]
    oof = np.zeros((len(df), len(cols)))
    best_eval_score_list = []

    for cv in cv_list:

        LOGGER.info('# ===============================================================================')
        LOGGER.info(f'# Start CV: {cv}')
        LOGGER.info('# ===============================================================================')

        # tensorboard
        writer = SummaryWriter(log_dir=output_path)

        df_train = df[df.cv != cv].reset_index(drop=True)
        # df_train = df_train[df_train.StudyInstanceUID.isin(anot_index)].reset_index()
        df_val = df[df.cv == cv].reset_index(drop=True)
        val_index = df[df.cv == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = RanzcrDataset(df=df_train, image_size=img_size,
                                      image_folder=img_dir, from_image_folder=True,
                                      transform=train_transform, mode="train", clahe=clahe, mix=mix)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, num_workers=n_workers, drop_last=True)
        # plot sample image
        # plot_sample_images(train_dataset, sample_img_path, "train", normalize="imagenet")

        val_dataset = RanzcrCVCDataset5(df=df_val, image_size=img_size,
                                    image_folder=img_dir, from_image_folder=True,
                                    transform=val_transform, mode="train", clahe=clahe, mix=mix)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    pin_memory=False, num_workers=n_workers, drop_last=False)

        # plot_sample_images(val_dataset, sample_img_path, "val",  normalize="imagenet")

        # ==== INIT MODEL
        device = torch.device(device)

        # aux_params = dict(
        #     pooling='avg',  # one of 'avg', 'max'
        #     dropout=0.2,  # dropout ratio, default is None
        #     activation=None,  # activation function, default is None
        #     classes=len(cols),  # define number of output labels
        # )
        aux_params = None

        model = smp.Unet(
            encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            activation=None,
            aux_params=aux_params
        ).to(device)
        # if model_name == "resnet200d":
        #     model = RANZCRResNet200D().to(device)
        # else:
        #     model = Net(model_name).to(device)
        load_checkpoint = cfg["load_checkpoint"][cv]
        LOGGER.info("-" * 10)
        if os.path.exists(load_checkpoint):
            weight = torch.load(load_checkpoint, map_location=device)
            if "exp" in load_checkpoint:
                model.load_state_dict(weight["state_dict"])
            else:
                model.load_state_dict(weight)
            LOGGER.info(f"Successfully loaded model, model path: {load_checkpoint}")
        else:
            LOGGER.info(f"Training from scratch..")
        LOGGER.info("-" * 10)

        optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)


        criterion = [
            nn.BCEWithLogitsLoss(reduction='none'),
            SymmetricLovaszLoss()
        ]
        scaler = GradScaler()

        # ==== TRAIN LOOP

        best = 99
        best_epoch = 0
        early_stopping_cnt = 0

        # ==== EVAL LOOP
        eval_time = time.time()
        model.eval()
        torch.set_grad_enabled(False)
        losses_eval = []
        eval_correct = 0
        pred_list = []
        targets_list = []

        sample_mask_pred = None
        losses_label = []
        losses_mask = []

        oof_list = []

        progress_bar_eval = tqdm(val_dataloader)
        for step_eval, data in enumerate(progress_bar_eval):
            if debug:
                if step_eval == 2:
                    break
            loss, pred, targets, pred_labels, loss_label, loss_mask, pred_mask, id_list\
                                                = forward2(data, model, device, criterion)
            losses_eval.append(loss.item())

            progress_bar_eval_text = f"Running EVAL, loss: {loss.item()} loss(avg): {np.mean(losses_eval)}"
            progress_bar_eval.set_description(progress_bar_eval_text)

            for s in range(len(pred_mask)):
                m = ((np.array(pred_mask[s]) > 0.5)*255)[0].astype(np.uint8)
                # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2),np.uint8))
                cv2.imwrite(os.path.join(save_mask, f"{id_list[s]}.png"), m)

    LOGGER.info('#'*20)
    LOGGER.info('# End')
    LOGGER.info('#'*20)