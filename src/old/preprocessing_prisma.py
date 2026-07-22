import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import scipy.interpolate
from matplotlib.ticker import MaxNLocator
import time
import gc
import imageio
import h5py


def afficher_image_prisma(chemin_h5):
    # 1. Ouvrir le fichier HDF5 en mode lecture ('r')
    with h5py.File(chemin_h5, "r") as f:

        def inspecter(name):
            print(name)
        

        print(f['HDFEOS/SWATHS/PRS_L1_PCO/Geolocation Fields/Latitude'][0,:])
        print(f['HDFEOS/SWATHS/PRS_L1_PCO/Geolocation Fields/Longitude'][0,:])

        
        longueurs_onde_swir = f["KDP_AUX/Cw_Swir_Matrix"]
        print(
        longueurs_onde_swir[0,:]
        )  

        longueurs_onde_vnir = f["KDP_AUX/Cw_Vnir_Matrix"]
        print(
        longueurs_onde_vnir[0,:]
        ) 

        wl_vnir = np.array( [981.6649 , 971.3399  ,960.9087 , 949.432  , 938.33093, 928.1533,
 917.82996 ,907.2473 , 896.5753  ,885.9016 , 875.2759 , 864.522 ,   853.74316,
 843.03064 , 832.3985 , 821.8129 , 811.2137 , 800.62103 , 790.0769 , 779.68396,
 769.30316 , 758.8678 , 748.54456  , 738.26 ,   728.1448 , 718.08594 , 707.88965,
 698.24664 , 688.32904 , 678.5385 ,  668.91504 , 659.42267 , 649.9523 ,  640.48737,
 631.5421 , 622.49066 , 613.62775  ,604.7203  , 595.9254 ,  587.33124 , 578.90704,
 570.5919 ,  562.3614  , 554.2172  , 546.1432 , 538.2549  , 530.4708 , 522.69,
 515.0587  , 507.49118 , 499.9806  , 492.52676 , 485.213   , 477.9621  , 470.72733,
 463.49054 , 456.16    , 448.75095 , 441.3433  , 433.99976 , 426.5187  , 418.70532,
 410.10828])


        f.visit(inspecter)


        chemin_vnir = "HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"
        vnir_cube = f[chemin_vnir]

        print(f"Structure du cube VNIR : {vnir_cube.shape}")
        target_r = 650.0
        target_g = 550.0
        target_b = 470.0


        idx_r = np.abs(wl_vnir - target_r).argmin()
        idx_g = np.abs(wl_vnir - target_g).argmin()
        idx_b = np.abs(wl_vnir - target_b).argmin()

        print(f"Vrais indices trouvés -> Rouge: {idx_r}, Vert: {idx_g}, Bleu: {idx_b}")



       

  
        r = vnir_cube[:,idx_r, :]
        g = vnir_cube[:,idx_g, :]
        b = vnir_cube[:,idx_b, :]
    img_rgb = np.dstack((r, g, b))


    img_normalisee = np.zeros_like(img_rgb, dtype=np.float32)
    for i in range(3):
        bande = img_rgb[:, :, i]
        p2, p98 = np.percentile(bande, (2, 98))
        # On évite la division par zéro et on scale entre 0 et 1
        if p98 - p2 > 0:
            img_normalisee[:, :, i] = np.clip((bande - p2) / (p98 - p2), 0, 1)
    
    frames = []
    _,total_bandes,_=vnir_cube.shape
   
    for i in range(0, total_bandes, 1):
     
            
            # Calcul des percentiles pour éliminer les valeurs aberrantes de CETTE bande
        p2, p98 = np.percentile(bande, (2, 98))
            
        if p98 > p2:
                # On encadre les valeurs pour éliminer le bruit
            bande = np.clip(bande, p2, p98)
                # Étalement parfait de la dynamique entre 0 et 255
            bande = ((bande - p2) / (p98 - p2) * 255).astype(np.uint8)
        else:
            bande = np.zeros_like(bande, dtype=np.uint8)
                
        frames.append(bande)

    # 5. Affichage avec Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_normalisee)
    plt.title("Image Satellite  ")
    plt.savefig("Image prisma")
    plt.axis("off")  # Masquer les axes de pixels
    plt.show()

afficher_image_prisma('/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/Prisma/PRS_L1_STD_OFFL_20210312174127_20210312174132_0001 (10).he5')