# Multimodal-change-detection-for-remote-sensing-images

This repository contains the official implementation of our project on multimodal change detection between Hyperspectral (HSI) and Multispectral (MSI) images. 

Our approach avoids naive direct comparison by first projecting the MSI into the HSI domain, and then explicitly estimating the pixel-wise reconstruction uncertainty to guide the downstream change detection process.

## Project Overview

The pipeline is composed of two main deep learning networks:

1. **MSI-to-HSI Reconstruction (U-Net):** An asymmetric gradual expansion U-Net that predicts a high-spectral-resolution image (230 bands) from a multispectral input (12 bands, Sentinel-2B).
2. **Uncertainty Estimation (NAFNet):** A network based on NAFBlocks. It takes both the real MSI and the simulated HSI to predict the spatial-spectral aleatoric uncertainty map (estimating the Laplace scale parameter via $\mathcal{L}_1$ loss). This uncertainty map helps distinguish true physical ground changes from synthetic reconstruction artifacts.

## Dataset
The models are trained using synthetic MSI-HSI pairs generated from the [MUMUCD dataset](https://zenodo.org/records/10674011). The MSI data is simulated using the Sentinel-2B Spectral Response Function (SRF).