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


def ousm_loss(error, k=2):
    # ousm, drop large k sample
    bs = error.shape[0]
    if len(error.shape) == 2:
        error = error.mean(1)
    _, idxs = error.topk(bs - k, largest=False)
    error = error.index_select(0, idxs)
    return error


# change targets, https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/186492
def forward(data, model, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    targets = data["target"].to(device)
    pred = model(inputs)
    pred_labels = torch.argmax(pred, dim=1)

    loss = criterion(pred, targets)
    loss = ousm_loss(loss, 3).mean()

    n_correct = (torch.argmax(pred, dim=1) == targets).cpu().detach().numpy().sum().item()

    batch_cm = confusion_matrix(targets.cpu().numpy(), pred_labels.cpu().numpy(),
                                labels=list(range(5)))

    return loss, pred, n_correct, batch_cm

#
# def get_train_transforms(image_size):
#     return albumentations.Compose([
#         albumentations.Transpose(p=0.5),
#         albumentations.VerticalFlip(p=0.5),
#         albumentations.HorizontalFlip(p=0.5),
#         albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
#                                                 brightness_by_max=False, p=0.5),
#         albumentations.Blur(blur_limit=7, p=0.5),
#         # albumentations.HueSaturationValue(p=0.5),
#         albumentations.CenterCrop(540, 540, p=1),
#         albumentations.Resize(image_size, image_size),
#         # albumentations.RandomResizedCrop(height=image_size, width=image_size, scale=(0.08, 1)),
#         albumentations.CoarseDropout(max_holes=3, max_height=50, max_width=50),
#         ToTensorV2()
#     ])
#
#
# def get_val_transforms(image_size):
#     return albumentations.Compose([
#         albumentations.CenterCrop(540, 540, p=1),
#         albumentations.Resize(image_size, image_size),
#         # albumentations.RandomResizedCrop(height=image_size, width=image_size, scale=(0.08, 1)),
#         ToTensorV2()
#     ], p=1.0)



def get_train_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size, p=1),
        albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=(1., 3.), rotate_limit=0, p=0.2),
        albumentations.RandomCrop(512, 512, p=1),
        albumentations.Blur(blur_limit=10, p=0.3),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
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

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)

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

        # tensorboard
        writer = SummaryWriter(log_dir=output_path)

        df_train = df[df.cv != cv].reset_index(drop=True)
        df_val = df[df.cv == cv].reset_index(drop=True)
        val_index = df[df.cv == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = CassavaDataset(df=df_train, image_size=img_size,
                                       image_folder="/data/train_images",
                                       transform=train_transform, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, num_workers=n_workers, drop_last=True)
        # plot sample image
        plot_sample_images(train_dataset, sample_img_path, "train")

        val_dataset = CassavaDataset(df=df_val, image_size=img_size,
                                     image_folder="/data/train_images",
                                     transform=val_transform, mode="train")
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    pin_memory=False, num_workers=n_workers, drop_last=False)

        plot_sample_images(val_dataset, sample_img_path, "val")


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

        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=1e-6)

        # criterion = LabelSmoothingCrossEntropy()
        criterion = nn.CrossEntropyLoss(reduction='none')
        scaler = GradScaler()

        # ==== TRAIN LOOP

        best = -1
        best_epoch = 0

        for e in range(1, n_epochs+1):
            losses_train = []
            train_correct = 0

            train_time = time.time()
            LOGGER.info("")
            LOGGER.info("+"*30)
            LOGGER.info(f"+++++  Epoch {e}")
            LOGGER.info("+"*30)
            LOGGER.info("")
            progress_bar = tqdm(train_dataloader)

            model.train()
            torch.set_grad_enabled(True)

            for step_train, data in enumerate(progress_bar):
                if debug:
                    if step_train == 2:
                        break
                with autocast():
                    loss, pred, n_correct, _ = forward(data, model, device, criterion)

                # Backward pass
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses_train.append(loss.item())
                train_correct += n_correct
                progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

            LOGGER.info(f"Train loss: {np.mean(losses_train)}")
            LOGGER.info(f"Train Acc: {train_correct / len(df_train)}")
            LOGGER.info(f"Train time: {(time.time() - train_time) / 60:.3f} min")

            writer.add_scalar(f"Loss/train_cv{cv}", np.mean(losses_train), global_step=e)
            writer.add_scalar(f"Acc/train_cv{cv}", train_correct / len(df_train), global_step=e)

            # ==== EVAL LOOP
            eval_time = time.time()
            model.eval()
            torch.set_grad_enabled(False)
            losses_eval = []
            eval_correct = 0
            pred_list = []
            total_cm = np.zeros((5, 5))

            progress_bar_eval = tqdm(val_dataloader)
            for step_eval, data in enumerate(progress_bar_eval):
                if debug:
                    if step_eval == 2:
                        break
                loss, pred, n_correct, batch_cm = forward(data, model, device, criterion)
                losses_eval.append(loss.item())
                eval_correct += n_correct
                total_cm += batch_cm
                pred_list.extend(pred.cpu().numpy().tolist())
                progress_bar_eval_text = f"Running EVAL, loss: {loss.item()} loss(avg): {np.mean(losses_eval)}"
                progress_bar_eval.set_description(progress_bar_eval_text)

            print(np.array(pred_list).shape)
            # scheduler
            scheduler.step(e)

            LOGGER.info(f"Val loss: {np.mean(losses_eval)}")
            LOGGER.info(f"Val Acc: {eval_correct / len(df_val)}")
            LOGGER.info(f"Val time: {(time.time() - eval_time) / 60:.3f} min")

            writer.add_scalar(f"Loss/eval_cv{cv}", np.mean(losses_eval), global_step=e)
            writer.add_scalar(f"Acc/eval_cv{cv}", eval_correct / len(df_val), global_step=e)

            LOGGER.info('Saving model ...')
            model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint{e}.pth")

            torch.save({
                "state_dict": model.state_dict(),
            }, model_save_path)

            acc = eval_correct / len(df_val)

            if best < acc:
                LOGGER.info(f'Best score update: {best:.5f} --> {acc:.5f}')
                best = acc
                best_epoch = e

                # save confusion matrix
                plt.figure()
                plt.title(f'Epoch: {e}, ACC: {eval_correct / len(df_val):.4f}, Loss: {np.mean(losses_eval):.4f}')
                sns.heatmap(total_cm.astype(np.int), annot=True, fmt="d")
                # writer.add_figure(f"Eval/confusion_matrix_cv{cv}", fig, global_step=e)
                plt.savefig(os.path.join(plot_path, f"cm_cv{cv}_e{e}.png"))
                plt.close()

                try:
                    if debug:
                        oof[batch_size*2] = np.array(pred_list)
                    else:
                        oof[val_index] = np.array(pred_list)

                except Exception as error:
                    LOGGER.info(error)

            LOGGER.info('-' * 20)
            LOGGER.info(f'Best val score: {best}, at epoch {best_epoch} cv{cv}')
            LOGGER.info('-' * 20)

        writer.close()

    #######################################
    ## Save oof
    #######################################

    LOGGER.info('-' * 20)
    LOGGER.info(f'Oof score: {(oof.argmax(1) == df.label).sum() / len(df)}, at epoch {e}')
    LOGGER.info('-' * 20)

    np.save(os.path.join(oof_path, "oof"), oof)
