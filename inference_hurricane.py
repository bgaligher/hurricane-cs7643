import numpy as np
import os
import torch
import cv2
import argparse
import yaml

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import path, listdir
from tqdm import tqdm
from collections import OrderedDict

from utils.utils import *
from models.hurricane import HurricaneModel, HurricaneAddModel, HurricaneCatModel, HurricaneDOModel, HurricaneLoRAModel

sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
seed = "42"

holdout_path = os.path.join("data_wind", "wind_hold")

# Create data loader
hold_files = []
for f in sorted(listdir(path.join(holdout_path, 'images'))):
    if '_pre_disaster.png' in f:
        hold_files.append(path.join(holdout_path, 'images', f))

hold_idxs = np.arange(len(hold_files))

class HoldoutData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = hold_files[_idx]

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
    
'''
F1 - Harmonic Mean
F1_0 - No Damage
F1_1 - Minor
F1_2 - Major
F1_3 - Destroyed
'''

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
    print("Final Score: {}, Dice: {}, F1 (Harmonic Mean): {}, F1 (None): {}, F1 (Minor): {}, F1 (Major): {}, F1 (Destroyed): {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    return (sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Hurricane Model - Inference")
    parser.add_argument('--job_id', type=str, default='0', help='SLURM job ID')
    args = parser.parse_args()
    job_id = args.job_id

    # find checkpoint file based on job ID
    checkpoint_dir, checkpoint_file = "weights", None
    for root, dir, files in os.walk(checkpoint_dir):
        for file in files:
            if job_id in file:
                checkpoint_file = file

    weights_path = os.path.join(checkpoint_dir,checkpoint_file)

    # Move model to device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load checkpoint
    checkpoint = torch.load(weights_path,
                            map_location=device,
                            weights_only=False)

    state_dict = checkpoint['state_dict']

    if 'hurricane_c' in checkpoint_file:
        model = HurricaneCatModel(checkpoint_path=sam_checkpoint).to(device)
    elif 'hurricane_a' in checkpoint_file:
        model = HurricaneAddModel(checkpoint_path=sam_checkpoint).to(device)
    elif 'hurricane_do' in checkpoint_file:
        model = HurricaneDOModel(checkpoint_path=sam_checkpoint).to(device)
    elif 'hurricane_lora' in checkpoint_file:
        model = HurricaneLoRAModel(checkpoint_path=sam_checkpoint).to(device)
    else:
        model = HurricaneModel(checkpoint_path=sam_checkpoint).to(device)
    
    # Remove 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    model.to(device)

    # Create DataLoader
    inference_data = HoldoutData(hold_idxs)
    inference_data_loader = DataLoader(inference_data, 
                                    batch_size=1,
                                    shuffle=False)

    # Inference
    print(f"Running inference for Job ID: {job_id}...")
    val_data = validate(model, 
                        data_loader=inference_data_loader,
                        device=device)
