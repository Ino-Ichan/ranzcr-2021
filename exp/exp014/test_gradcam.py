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

import albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import argparse
import os, sys, yaml

sys.path.append('./')
from src.logger import setup_logger, LOGGER
from src.models import Net
from src.losses import LabelSmoothingCrossEntropy
from src.dataset import CassavaDataset
from src.utils import plot_sample_images
from src.augmix import RandomAugMix
import warnings

import time
from contextlib import contextmanager

## gradcam
from torchvision.utils import make_grid
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# change targets, https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/186492
def forward(data, model, gradcam, gradcam_pp, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    targets = data["target"].to(device)

    pred = model(inputs)
    pred_labels = torch.argmax(pred, dim=1)

    loss = criterion(pred, targets)
    loss = loss.mean()

    # gradcam
    original_imgs = inputs[0]
    images = []

    mask, _ = gradcam(inputs)
    heatmap, result = visualize_cam(mask, original_imgs)

    mask_pp, _ = gradcam_pp(inputs)
    heatmap_pp, result_pp = visualize_cam(mask_pp, original_imgs)

    images.extend([original_imgs.cpu(), heatmap, result, heatmap_pp, result_pp])
    grid_image = make_grid(images, nrow=5)

    return loss, pred, grid_image




def get_train_transforms(image_size):
    return albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Resize(image_size, image_size),
        RandomAugMix(severity=5, width=4, depth=4, alpha=1., always_apply=True, p=0.9),
        albumentations.CoarseDropout(max_holes=3, max_height=50, max_width=50),
        ToTensorV2()
    ], p=1.0)


def get_val_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        ToTensorV2()
    ], p=1.0)



if __name__ == "__main__":
    print('Start!!!')
    os.environ["L5KIT_DATA_FOLDER"] = "/data/"
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
    # seed_everythin
    seed_torch()

    # output
    exp_name = cfg["exp_name"]  # os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join("/workspace/output", exp_name)
    # path
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    oof_path = output_path + "/oof"
    sample_img_path = output_path + "/sample_img"
    gradcam_path = output_path + "/gradcam"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)
    os.makedirs(gradcam_path, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log_gradcam.txt")
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
    transform = cfg["transform"]
    hold_out = cfg["hold_out"]

    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg["df_train_path"])

    cv_list = hold_out if hold_out else [0, 1, 2, 3, 4]
    oof = np.zeros((len(df), 5))

    for cv in cv_list:

        LOGGER.info('# ===============================================================================')
        LOGGER.info(f'# Start CV: {cv}')
        LOGGER.info('# ===============================================================================')

        df_train = df[df.cv != cv].reset_index(drop=True)
        df_val = df[df.cv == cv].reset_index(drop=True)
        val_index = df[df.cv == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        val_transform = get_val_transforms(img_size)

        val_dataset = CassavaDataset(df=df_val, image_size=img_size,
                                     image_folder="/data/train_images",
                                     transform=val_transform, mode="train")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    pin_memory=False, num_workers=n_workers, drop_last=False)

        plot_sample_images(val_dataset, sample_img_path, "val")

        criterion = nn.CrossEntropyLoss(reduction="none")


        # ==== INIT MODEL
        device = torch.device(device)
        model = Net(model_name).to(device)
        load_checkpoint = cfg["load_checkpoint"][cv]
        LOGGER.info("-"*10)
        if os.path.exists(load_checkpoint):
            weight = torch.load(load_checkpoint)["state_dict"]
            model.load_state_dict(weight)
            LOGGER.info(f"Successfully loaded model, model path: {load_checkpoint}")
        else:
            LOGGER.info(f"Training from scratch..")
        LOGGER.info("-" * 10)

        # ==== EVAL LOOP
        eval_time = time.time()
        model.eval()
        # torch.set_grad_enabled(False)

        # ==================================================
        # Set Grad Cam
        # ==================================================
        # target_layer = model.model.layer4[2].act3
        target_layer = model.model.layer4
        gradcam = GradCAM(model, target_layer)
        gradcam_pp = GradCAMpp(model, target_layer)

        progress_bar_eval = tqdm(val_dataloader)
        for step_eval, data in enumerate(progress_bar_eval):
            if debug:
                if step_eval == 2:
                    break
            row = df_val.iloc[step_eval]
            loss, pred, grid_image = forward(data, model, gradcam, gradcam_pp, device, criterion)
            plt.figure(figsize=(12, 5))
            plt.imshow(grid_image.cpu().numpy().transpose(1, 2, 0))
            plt.title(f"image_id: {row.image_id}, label: {data['target'].item()}\n"
                      f"pred: {pred.softmax(1).detach().cpu().numpy()}, loss: {loss}")
            plt.savefig(os.path.join(gradcam_path, f"label{data['target'].item()}_loss{loss:.5f}_pred{pred.argmax(1).item()}_{row.image_id}"))
            plt.close()

    #######################################
    ## Save oof
    #######################################

    # LOGGER.info('-' * 20)
    # LOGGER.info(f'Oof score: {(oof.argmax(1) == df.label).sum() / len(df)}}')
    # LOGGER.info('-' * 20)
    #
    # np.save(os.path.join(oof_path, "oof"), oof)
