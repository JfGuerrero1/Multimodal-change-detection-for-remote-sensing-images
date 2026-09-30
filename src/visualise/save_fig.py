import os
import warnings
from typing import Any, Callable, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# Gestion des axes d'images
# ==========================================

def manage_image_axis(im: np.ndarray, chan_axis: bool = True, batch_axis: bool = False) -> np.ndarray:
    """Gère l'ajout ou la suppression des axes de canaux et de batch pour les images."""
    assert len(im.shape) != 1, f"Input of shape {im.shape} is not an image (it must be at least 2D)."

    if chan_axis and (len(im.shape) <= 2):
        im = im[..., np.newaxis]
    elif (not chan_axis) and (len(im.shape) >= 3):
        if im.shape[-1] != 1:
            warnings.warn("/!\\ Removing the channel axis will lead to a loss of data...")
        im = im[..., 0]

    if batch_axis and (len(im.shape) <= 3):
        im = im[np.newaxis, ...]
    elif (not batch_axis) and (len(im.shape) >= 4):
        if im.shape[0] != 1:
            warnings.warn("/!\\ Removing the batch axis will lead to a loss of data...")
        im = im[0, ...]

    return im


# ==========================================
# Fonctions de normalisation
# ==========================================

def get_meanstd(arr: np.ndarray, mstd: float = 3.0) -> float:
    return float(np.mean(arr)) + mstd * float(np.std(arr))


def minmax_norm(arr: np.ndarray, mini: float = 0.0, maxi: float = 1.0) -> np.ndarray:
    return (arr - mini) / (maxi - mini)


def minmax_denorm(arr: np.ndarray, mini: float = 0.0, maxi: float = 1.0) -> np.ndarray:
    return arr * (maxi - mini) + mini


def minmax_normclip(arr: np.ndarray, mini: float = 0.0, maxi: float = 1.0) -> np.ndarray:
    return np.clip(minmax_norm(arr, mini=mini, maxi=maxi), 0, 1)


def multip_clip(arr: np.ndarray, mult: float = 1.0) -> np.ndarray:
    return np.clip(arr * mult, 0.0, 1.0)


NORM_CONSTS = {
    "m": -1.429329123112601, "M": 10.089038980848645,
    "d": 0, "D": 1.2,
    "g": 0, "G": 4.5,
    "r": 1.2533141373155001, "R": 0.6551363775620336
}


def get_norm_consts() -> dict:
    return NORM_CONSTS


def get_norm_const(key: str) -> float:
    return NORM_CONSTS[key]


def norm_meth_cmap_from_name(
    name: str, do_norm: bool = True, do_clip: bool = True
) -> Union[tuple[str, str], tuple[list[str], list[str]]]:
    if do_norm and do_clip:
        pref_meth = "normclip_"
    elif do_clip:
        pref_meth = "clip_"
    elif do_norm:
        pref_meth = "norm_"
    else:
        return "", "grey"

    if name.lower() == "mapbiomasalerta":
        return "", "grey"
    elif ("histo-" in name) and (("_norm_by_" in name) or ("_whiten_by_" in name)):
        return "clip_-4.4172_4.4172", "grey"
    elif name.startswith("th_") or name.startswith("th-"):
        return "", "grey"
    elif "pxloss_" in name:
        return pref_meth + "quant_0.05_0.95", "turbo"
    elif ("log_" in name) and not ("_log_" in name) and not ("log_cd" in name):
        return pref_meth + "1m_1M", "grey"
    elif "spectrum" in name:
        return pref_meth + "quant_0_0.95", "grey"
    elif "autocorr" in name:
        return pref_meth + "-1.07_1.07", "bwr"
    elif "map_alpha" in name:
        return pref_meth + "minmax", "jet"
    elif "delta" in name:
        return (pref_meth + "d_2D", "turbo") if "noisy" in name else (pref_meth + "d_D", "turbo")
    elif "fus_2unc" in name:
        return pref_meth + "d_D", "turbo"
    elif "gamma" in name:
        return pref_meth + "g_G", "turbo"
    elif ("_norm_by_" in name) and ("abs_" in name):
        return pref_meth + "g_G", "turbo"
    elif "_norm_by_" in name or "_whiten_by_" in name:
        return pref_meth + "-1.96_1.96", "viridis"
    elif "var_decorr_ponds" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "quant_0.025_0.975", pref_meth + "0_0.8"], ["plasma", "cividis", "pink"]
    elif "var_decorr" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "quant_0.025_0.975"], ["plasma", "cividis"]
    elif "var_whit_ponds" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "-1.96_1.96", pref_meth + "0_1"], ["plasma", "viridis", "pink"]
    elif "var_whit" in name:
        return [pref_meth + "quant_0_0.975", pref_meth + "-1.96_1.96"], ["plasma", "viridis"]
    elif ("var_law" in name) or ("vardlr" in name) or ("covmat_law-" in name):
        return pref_meth + "d_D", "plasma"
    elif "expect_abs_diff_log_ref" in name:
        return pref_meth + "d_1.5D", "turbo"
    elif "filt" in name:
        return pref_meth + "-1_1", "viridis"
    elif "ad_conf_pred" in name:
        return pref_meth + "g_G", "turbo"
    elif "abs_diff_log_ref" in name or "abs_diff_log_bm3d_ref" in name:
        return pref_meth + "d_D", "cividis"
    elif ("diff_log_ref" in name) or ("dlr" in name) or ("diff_log_bm3d_ref" in name):
        return pref_meth + "-1D_D", "cividis"
    elif "residual_noise" in name:
        return pref_meth + "0_r+2*R", "grey"
    elif "noisy" in name:
        return pref_meth + "meanstd_3", "grey"
    elif "denoised" in name:
        return pref_meth + "meanstd_2", "grey"
    elif "changemaskv2" in name:
        return [pref_meth + "quant_0.01_0.99", pref_meth + "quant_0.01_0.99"], ["PuOr_r", "PiYG"]
    elif "changemask" in name:
        return [pref_meth + "0_1", pref_meth + "-1.96_1.96"], ["PuOr_r", "viridis"]
    elif "log_cd_as_uncert_determv2" in name:
        return pref_meth + "-1.96_1.96", "cividis"
    elif "log_cd_as_uncert_determ" in name:
        return pref_meth + "-1D_D", "cividis"
    elif "cd_as_uncert_determ" in name:
        return pref_meth + "quant_0_0.99", "afmhot"
    elif "cd_as_uncert_probab" in name:
        return pref_meth + "d_0.3D", "plasma"
    elif "gwn" in name:
        return pref_meth + "-1.96_1.96", "viridis"
    elif "gt" in name:
        return "", "grey"

    raise AssertionError(f"Unrecognized name '{name}' to determine the right normalization method.")


def get_extremums_by_method(
    im: np.ndarray, im4norm: np.ndarray = None, normalization_method: str = None
) -> tuple[float, float]:
    consts_norm = get_norm_consts()
    keys_const = consts_norm.keys()

    if im4norm is None:
        im4norm = im.copy()

    sp_norm_meth = normalization_method.split("_")

    if sp_norm_meth[1] == "quant":
        mini = np.min(im4norm) if float(sp_norm_meth[2]) == 0 else np.quantile(im4norm, float(sp_norm_meth[2]))
        maxi = np.max(im4norm) if float(sp_norm_meth[3]) == 1 else np.quantile(im4norm, float(sp_norm_meth[3]))
    elif sp_norm_meth[1] == "minmax":
        mini = np.min(im4norm)
        maxi = np.max(im4norm)
    elif sp_norm_meth[1] == "meanstd":
        mini = 0
        maxi = get_meanstd(im4norm, float(sp_norm_meth[2]))
    else:
        minmax = [-np.inf, np.inf]
        for i in [1, 2]:
            if sp_norm_meth[i][-1] in keys_const:
                if len(sp_norm_meth[i]) == 1:
                    minmax[i - 1] = consts_norm[sp_norm_meth[i]]
                else:
                    try:
                        minmax[i - 1] = consts_norm[sp_norm_meth[i][-1]] * float(sp_norm_meth[i][:-1])
                    except:
                        str2eval = sp_norm_meth[i]
                        for k in keys_const:
                            str2eval = str2eval.replace(k, str(consts_norm[k]))
                        minmax[i - 1] = eval(str2eval)
            else:
                minmax[i - 1] = float(sp_norm_meth[i])
        mini, maxi = minmax[0], minmax[1]

    return mini, maxi


# ==========================================
# Gestion des Colormaps
# ==========================================

def get_list_custom_cmaps() -> list[str]:
    return ["bwr", "log_PRGn"]


def get_custom_cmap(name_cmap: str, cmap_length: int = 256) -> np.ndarray:
    arng = ((np.arange(cmap_length) / (cmap_length - 1)) - 0.5) * 2
    cust_cmap = np.empty((*arng.shape[:-1], 3))

    if name_cmap == "bwr":
        norm_im = cust_cmap * 2 - 1
        cust_cmap[..., 2] = 1 + np.clip(norm_im, -1, 0)
        cust_cmap[..., 0] = 1 - np.clip(norm_im, 0, 1)
        cust_cmap[..., 1] = 1 + np.clip(norm_im, -1, 0) - np.clip(norm_im, 0, 1)
    else:
        raise ValueError(f"Unknown custom color map name '{name_cmap}'.")

    return cust_cmap


def apply_custom_cmap(norm_im_gray: np.ndarray, name_cmap: str) -> np.ndarray:
    if norm_im_gray.shape[-1] != 1:
        warnings.warn(f"Please provide a grayscale image. Adding a channel to shape: {norm_im_gray.shape}")
        norm_im_gray = norm_im_gray[..., np.newaxis]

    new_im = np.empty((*norm_im_gray.shape[:-1], 3))

    if name_cmap == "bwr":
        norm_im_gray = norm_im_gray * 2 - 1
        new_im[..., 2] = 1 + np.clip(norm_im_gray[..., 0], -1, 0)
        new_im[..., 0] = 1 - np.clip(norm_im_gray[..., 0], 0, 1)
        new_im[..., 1] = 1 + np.clip(norm_im_gray[..., 0], -1, 0) - np.clip(norm_im_gray[..., 0], 0, 1)
    else:
        try:
            c = np.linspace(0, 1, 256)
            c = plt.get_cmap(name_cmap)(c)[:, :3][:, ::-1]
            norm_im_gray = (np.clip(norm_im_gray, 0, 1) * 255).astype(np.uint8)
            new_im = [c[norm_im_gray, i] for i in range(3)]
            new_im = np.concatenate(new_im, axis=-1)
        except Exception as e:
            raise ValueError(f"Unknown custom color map name '{name_cmap}'. Error: {e}")

    return new_im


def cmap2apply_from_name(cmap_name: str) -> tuple[Any, bool]:
    is_custom_cmap = False
    if cmap_name is None or cmap_name == "grey":
        return None, False

    cmap_mapping = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "cividis": cv2.COLORMAP_CIVIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "pink": cv2.COLORMAP_PINK,
        "spring": cv2.COLORMAP_SPRING,
        "cool": cv2.COLORMAP_COOL,
        "hot": cv2.COLORMAP_HOT,
        "ocean": cv2.COLORMAP_OCEAN,
        "twilight": cv2.COLORMAP_TWILIGHT,
        "twilight_shifted": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "hsv": cv2.COLORMAP_HSV
    }

    if cmap_name.lower() in cmap_mapping:
        return cmap_mapping[cmap_name.lower()], False
    elif (cmap_name in get_list_custom_cmaps()) or (cmap_name in plt.colormaps()):
        return cmap_name, True
    else:
        raise ValueError(f"Unknown color map '{cmap_name}' for normalization.")


def normalize_by_method(
    im: np.ndarray, im4norm: np.ndarray = None, normalization_method: str = None, out_cmap: str = "grey"
) -> np.ndarray:
    im = manage_image_axis(im, chan_axis=True, batch_axis=True)
    im_norm = im.copy()
    new_mini, new_maxi = None, None

    if isinstance(normalization_method, list):
        if isinstance(out_cmap, str):
            out_cmap = [out_cmap for _ in range(im.shape[-1])]
        else:
            out_cmap += [out_cmap[-1]] * (im.shape[-1] - len(out_cmap))

        l_im_norm = []
        for c in range(len(normalization_method)):
            sub_im = im[..., c:] if (c == len(normalization_method) - 1 and c < im.shape[-1] - 1) else im[..., c:c+1]
            if im4norm is None:
                sub_norm = None
            else:
                sub_norm = im4norm if im4norm.shape[-1] == 1 else im4norm[..., c:] if (c == len(normalization_method) - 1 and c < im.shape[-1] - 1) else im4norm[..., c:c+1]
            
            l_im_norm.append(normalize_by_method(sub_im, im4norm=sub_norm, normalization_method=normalization_method[c], out_cmap=out_cmap[c]))
        
        return np.concatenate(l_im_norm, axis=-1)

    elif (normalization_method is not None) and (normalization_method != ""):
        if im4norm is None:
            im4norm = im.copy()
        else:
            im4norm = manage_image_axis(im4norm, chan_axis=True, batch_axis=True)

        sp_norm_meth = normalization_method.split("_")
        method = sp_norm_meth[0]

        if method in ["clip", "norm", "normclip", "clipnorm"]:
            mini, maxi = get_extremums_by_method(im, im4norm=im4norm, normalization_method=normalization_method)
            if method == "clip":
                im_norm = np.clip(im, mini, maxi)
                new_mini, new_maxi = mini, maxi
            elif method == "norm":
                im_norm = minmax_norm(im, mini=mini, maxi=maxi)
                new_mini, new_maxi = 0, 1
            else:
                im_norm = minmax_normclip(im, mini=mini, maxi=maxi)
                new_mini, new_maxi = 0, 1
        elif method == "multip":
            im_norm = multip_clip(im, float(sp_norm_meth[1]))
        else:
            im_norm = im.copy()
            warnings.warn(f"[normalize_by_method] Unknown normalization method {method} from {normalization_method}.")

    if isinstance(out_cmap, list):
        out_cmap += [out_cmap[-1]] * (im.shape[-1] - len(out_cmap))
        l_cm2ap_iscust = [cmap2apply_from_name(cm) for cm in out_cmap]
        npdtype = im_norm.dtype

        lc_im_norm = []
        for c, (cmap2apply, is_custom_cmap) in enumerate(l_cm2ap_iscust):
            if cmap2apply is None:
                lc_im_norm.append(im_norm[..., c:c+1])
            else:
                nm_mini = new_mini if new_mini is not None else np.min(im_norm[..., c])
                nm_maxi = new_maxi if new_maxi is not None else np.max(im_norm[..., c])
                if not is_custom_cmap:
                    lc_im_norm.append(np.stack([minmax_denorm(
                        cv2.applyColorMap((minmax_norm(im_norm[d, ..., c], nm_mini, nm_maxi) * 255).astype(np.uint8), cmap2apply).astype(npdtype) / 255, nm_mini, nm_maxi)
                        for d in range(im_norm.shape[0])], axis=0))
                else:
                    lc_im_norm.append(minmax_denorm(apply_custom_cmap(minmax_norm(im_norm[..., c:c+1], nm_mini, nm_maxi), cmap2apply).astype(npdtype), nm_mini, nm_maxi))
        im_norm = np.concatenate(lc_im_norm, axis=-1)
    else:
        cmap2apply, is_custom_cmap = cmap2apply_from_name(out_cmap)
        if cmap2apply is not None:
            npdtype = im_norm.dtype
            nm_mini = new_mini if new_mini is not None else np.min(im_norm)
            nm_maxi = new_maxi if new_maxi is not None else np.max(im_norm)
            if not is_custom_cmap:
                im_norm = np.stack([np.concatenate([minmax_denorm(
                    cv2.applyColorMap((minmax_norm(im_norm[d, ..., c], nm_mini, nm_maxi) * 255).astype(np.uint8), cmap2apply).astype(npdtype) / 255, nm_mini, nm_maxi)
                    for c in range(im_norm.shape[3])], axis=2) for d in range(im_norm.shape[0])], axis=0)
            else:
                im_norm = np.concatenate([minmax_denorm(apply_custom_cmap(minmax_norm(im_norm[..., c:c+1], nm_mini, nm_maxi), cmap2apply).astype(npdtype), nm_mini, nm_maxi) for c in range(im_norm.shape[3])], axis=3)

    return im_norm


# ==========================================
# Fonctions de sauvegarde
# ==========================================

def save_data_with_ext(
    im, path, ext, get_sub_window=False, id_sub_window=0, name4win=None,
    save3Dnumpy=False, save_info_npy=False, im4norm=None, norm_method=None,
    m255=True, save_cmap="grey", isrgb=False, nb_d_max=256, nb_c_max=256, c2save=None
):
    im = manage_image_axis(im, chan_axis=True, batch_axis=not (save3Dnumpy and ext == ".npy"))

    if ext in [".npy", ".np"]:
        np.save(os.path.splitext(path)[0], im)
        if save_info_npy:
            with open(os.path.splitext(path)[0] + ".inf", "w") as f:
                f.write(" ".join([str(im.shape[i]) for i in [1, 2, 3, 0]]) + "\n")
                f.write(f"-type {im.dtype}")
    elif ext == ".png":
        store_data_png(im, path, im4norm=im4norm, norm_method=norm_method, m255=m255,
                       save_cmap=save_cmap, isrgb=isrgb, nb_d_max=nb_d_max, nb_c_max=nb_c_max, c2save=c2save)
    else:
        raise AssertionError(f"Unknown extension {ext} to save {path}")


def store_data_png(
    im, filepath, im4norm=None, norm_method=None, m255=True, save_cmap="grey",
    isrgb=False, nb_d_max=256, nb_c_max=256, c2save=None
):
    im = manage_image_axis(im, chan_axis=True, batch_axis=True)

    if c2save is None:
        c2save = [i for i in range(min(im.shape[-1], nb_c_max))]
    elif not isinstance(c2save, list):
        c2save = [c2save]
    
    nb_c_max = min(nb_c_max, len(c2save))
    nb_chan_beg = len(c2save)
    
    im = normalize_by_method(im[..., c2save], im4norm=im4norm, normalization_method=norm_method, out_cmap=save_cmap)
    
    nb_chan_end = im.shape[-1]
    nb_rgb = (nb_chan_end - nb_chan_beg) // 2
    nb_no_rgb = nb_chan_beg - nb_rgb

    if m255:
        im = im * 255

    filepath = os.path.splitext(filepath)[0]
    fpath = filepath.split("/")
    fold_path = "/".join(fpath[:-1])
    file_name = fpath[-1].split("_")
    file_name = [file_name[0], "_".join(file_name[1:])]
    if file_name[1] != "":
        file_name[1] = "_" + file_name[1]
    file_name[1] += ".png"

    if (min(im.shape[0], nb_d_max) == 1) and ((min(im.shape[3], nb_c_max) == 1) or ((isrgb or save_cmap != "grey") and min(im.shape[3], nb_c_max) == 3)):
        save_path_gen = lambda ch, da: filepath + ".png"
    elif min(im.shape[0], nb_d_max) == 1:
        save_path_gen = lambda ch, da: os.path.join(fold_path, f"{file_name[0]}_chan{c2save[ch]}{file_name[1]}")
    elif (min(im.shape[3], nb_c_max) == 1) or ((isrgb or save_cmap != "grey") and min(im.shape[3], nb_c_max) == 3):
        save_path_gen = lambda ch, da: os.path.join(fold_path, f"{file_name[0]}_date{da}{file_name[1]}")
    else:
        save_path_gen = lambda ch, da: os.path.join(fold_path, f"{file_name[0]}_date{da}_chan{c2save[ch]}{file_name[1]}")

    for d in range(min(im.shape[0], nb_d_max)):
        if not (isrgb or (save_cmap != "grey")):
            for c in range(min(im.shape[3], nb_c_max)):
                cv2.imwrite(save_path_gen(c, d), im[d, :, :, c], [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            if nb_rgb in [0, nb_chan_beg]:
                assert im.shape[3] % 3 == 0, f"Image of shape {im.shape} can not be a stack of RGB."
                for c in range(0, min(im.shape[3], 3 * nb_c_max), 3):
                    cv2.imwrite(save_path_gen(c // 3, d), im[d, :, :, c:c + 3], [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                for c in range(nb_no_rgb):
                    cv2.imwrite(save_path_gen(c, d), im[d, :, :, c], [cv2.IMWRITE_PNG_COMPRESSION, 6])
                for cc in range(0, nb_c_max - nb_no_rgb):
                    c_idx = nb_no_rgb + cc
                    crgb = nb_no_rgb + 3 * cc
                    cv2.imwrite(save_path_gen(c_idx, d), im[d, :, :, crgb:crgb + 3], [cv2.IMWRITE_PNG_COMPRESSION, 6])

