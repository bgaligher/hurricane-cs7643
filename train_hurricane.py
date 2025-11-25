"""
Original code from: https://github.com/DIUx-xView/xView2_first_place

Edits for Project:
    - Disabled elastic image augmentation
    - Changed image_shape to 1024x1024
    - Swapped out Apex for Autocast for training mixed-precision
    - Integrated usage of config.yaml files for hyperparamters
    - Changed model forward pass method to take in pre/post images

References:
    1. https://docs.pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
"""


import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from torch.cuda.amp import autocast, GradScaler

from xview2_1st_place_solution.adamw import AdamW
from xview2_1st_place_solution.losses import dice_round, ComboLoss

from tqdm import tqdm
import timeit
import cv2

# from imgaug import augmenters as iaa

from xview2_1st_place_solution.utils import *

from skimage.morphology import square, dilation

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import argparse
import yaml
from models.hurricane import HurricaneModel
from utils.config import Config

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

train_dirs = ['data/TEST']  # Update this path if you have more training data directories

models_folder = 'weights'

loc_folder = 'pred_loc_val'

input_shape = (1024, 1024)  # SAM requires 1024x1024 input


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(d, 'images', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        # self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)
                    
        if random.random() > 0.9:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)
            
        if random.random() > 0.6:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.2:
            crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk3[y0:y0+crop_size, x0:x0+crop_size].sum() * 5 + msk4[y0:y0+crop_size, x0:x0+crop_size].sum() * 2 + msk1[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
        msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size]
        msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]
        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)
            

        if random.random() > 0.985:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.985:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.985:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.985:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.98:
            if random.random() > 0.985:
                img = clahe(img)
            elif random.random() > 0.985:
                img = gauss_noise(img)
            elif random.random() > 0.985:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.98:
            if random.random() > 0.985:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.98:
            if random.random() > 0.985:
                img2 = clahe(img2)
            elif random.random() > 0.985:
                img2 = gauss_noise(img2)
            elif random.random() > 0.985:
                img2 = cv2.blur(img2, (3, 3))
        elif random.random() > 0.98:
            if random.random() > 0.985:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)

                
        # if random.random() > 0.983:
        #     el_det = self.elastic.to_deterministic()
        #     img = el_det.augment_image(img)

        # if random.random() > 0.983:
        #     el_det = self.elastic.to_deterministic()
        #     img2 = el_det.augment_image(img2)

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk[..., 0] = False
        msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 3] = dilation(msk[..., 3], square(5))
        msk[..., 4] = dilation(msk[..., 4], square(5))
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = True
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = preprocess_inputs(img)
        img2 = preprocess_inputs(img2)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'pre_img': img, 'post_img': img2, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        lbl_msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)
        
        img = preprocess_inputs(img)
        img2 = preprocess_inputs(img2)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img2 = torch.from_numpy(img2.transpose((2, 0, 1))).float()  # Fixed: was using img instead of img2
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'pre_img': img, 'post_img': img2, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample


def validate(net, data_loader, device):
    dices0 = []

    tp = np.zeros((4,))
    fp = np.zeros((4,))
    fn = np.zeros((4,))

    _thr = 0.3

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            pre_img = sample["pre_img"].to(device, non_blocking=True)
            post_img = sample["post_img"].to(device, non_blocking=True)
            msks = sample["msk"].numpy()
            out = net(pre_img, post_img)

            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy() > 0.3
            msk_damage_pred = torch.sigmoid(out).cpu().numpy()[:, 1:, ...]
            
            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

                targ = lbl_msk[j][msks[j, 0] > 0]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > _thr)
                pred = pred[msks[j, 0] > 0]
                for c in range(4):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()

    d0 = np.mean(dices0)

    f1_sc = np.zeros((4,))
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * d0 + 0.7 * f1
    print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    return sc


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch, device):
    model = model.eval()
    d = validate(model, data_loader=data_val, device=device)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler, device):
    losses = AverageMeter()
    losses1 = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        pre_imgs = sample["pre_img"].to(device, non_blocking=True)
        post_imgs = sample["post_img"].to(device, non_blocking=True)
        msks = sample["msk"].to(device, non_blocking=True)

        with torch.autocast(device_type=device):
            out = model(pre_imgs, post_imgs)

            loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
            loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
            loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])

            loss = 0.05 * loss0 + 0.2 * loss1 + 0.8 * loss2 + 0.7 * loss3 + 0.4 * loss4

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), pre_imgs.size(0))
        losses1.update(loss2.item(), pre_imgs.size(0)) #loss5

        dices.update(dice_sc, pre_imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); loss2 {loss1.val:.4f} ({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; loss2 {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Hurricane Model Trainer")
    parser.add_argument('--seed', type=int, default=42, help="random seed for reproducibility (default: 42)")
    parser.add_argument('--config_file', type=str, default='configs/config_hurricane.yaml', help="path to YAML config")
    parser.add_argument('--output_dir', type=str, default='output/hurricane', help="path to output directory")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
        config = Config(config_dict=config_dict)
    
    model_name = config.model.name
    batch_size = config.train.batch_size
    epochs = config.train.n_epochs
    workers = config.train.num_workers
    lr = config.optimizer.lr
    weight_decay = config.optimizer.weight_decay

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    
    seed = args.seed

    cudnn.benchmark = True

    snapshot_name = f'{model_name}{seed}_0'

    file_classes = []
    for fn in tqdm(all_files):
        fl = np.zeros((4,), dtype=bool)
        msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        for c in range(1, 5):
            fl[c-1] = c in msk1
        file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)

    np.random.seed(seed + 321)
    random.seed(seed + 321)

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if file_classes[i, 1:].max():
            train_idxs.append(i)
        if file_classes[i, 1:3].max():
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False)

    model = HurricaneModel().to(device)

    params = model.parameters()

    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    
    scaler = torch.amp.GradScaler()

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    # Only use DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 12.0}, per_image=False).to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    best_score = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for epoch in range(epochs):
        train_epoch(epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, scaler, device)
        if epoch % 2 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch, device)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))