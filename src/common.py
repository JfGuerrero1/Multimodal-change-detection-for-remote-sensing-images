import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from models import GradualExpansionUNet, GradualExpansionUNet_residual
import scipy.interpolate
from matplotlib.ticker import MaxNLocator,FixedLocator
import time
import gc
from skimage.metrics import structural_similarity as ssim_sk
torch.cuda.init()
torch.cuda.set_device(0)
from constants import DEFAULT_SRF_PATH,INTERP_MATRIX


from metrics import compute_sam_map,compute_mae,compute_ergas,compute_mrae,compute_ssim_multiband,compute_mse,compute_sam
SRF_MATRIX=np.load(DEFAULT_SRF_PATH)

def compute_scene_data(gt_path, input_path, model, model_name, is_simulated=False, is_residual=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model.to(device)
    model.eval()

    with xr.open_dataset(gt_path) as ds_gt:
        cube_gt = np.nan_to_num(ds_gt["sr"].values, nan=0.0)
 
    h, w, c = cube_gt.shape
 
    if is_simulated:

        gt_flat = cube_gt.reshape(-1, c)
        msi_flat = np.dot(gt_flat, SRF_MATRIX)
        cube_msi = msi_flat.reshape(h, w, SRF_MATRIX.shape[1])
        input_numpy = cube_msi.transpose(2, 0, 1)
        msi_label = "MSI Simulée"
    else:
        if input_path is None:
            raise ValueError("Le paramètre 'input_path' est obligatoire pour les données réelles.")
        with xr.open_dataset(input_path) as ds_input:
            input_raw = np.nan_to_num(ds_input["sr"].values, nan=0.0)
        cube_msi = input_raw
        input_numpy = input_raw.transpose(2, 0, 1)
        msi_label = "MSI Réelle"
 
    cube_interp = None
    interp_tensor = None

    if is_residual:
        interp_numpy = INTERP_MATRIX @ input_numpy.reshape(12, -1)
        interp_numpy = interp_numpy.reshape(c, h, w)
        interp_tensor = torch.from_numpy(interp_numpy).unsqueeze(0).to(device)
        cube_interp = interp_numpy.transpose(1, 2, 0)

    input_tensor = torch.from_numpy(input_numpy).float().unsqueeze(0).to(device)
    cube_res = None
    
    with torch.no_grad():
        if is_residual:
            pred = model(input_tensor, interp_tensor)
            res = pred - interp_tensor
            cube_res = res.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        else:
            pred = model(input_tensor)
            
    cube_predict = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img_mse = compute_mse(cube_gt, cube_predict)
    img_sam = compute_sam(cube_gt, cube_predict)
    img_mae = compute_mae(cube_gt, cube_predict)
    img_ergas=compute_ergas(cube_gt,cube_predict)
    img_ssim=compute_ssim_multiband(cube_gt,cube_predict)
 
    return {
        "cube_gt": cube_gt,
        "cube_msi": cube_msi,
        "cube_predict": cube_predict,
        "cube_interp": cube_interp,
        "cube_res": cube_res,
        "msi_label": msi_label,
        "is_residual": is_residual,
        "img_mse": img_mse,
        "img_sam": img_sam,
        "img_mae": img_mae,
        "img_ergas": img_ergas,
        "img_ssim": img_ssim,
        "model name": model_name
    }