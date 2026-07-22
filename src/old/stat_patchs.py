import os
import glob
import random
import numpy as np
import xarray as xr
import torch
import seaborn as sns

from pathlib import Path
import scipy
import matplotlib.pyplot as plt 
import json
import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'dataset'
CACHE_DIR = DATA_DIR / 'patches_caches'

# Configuration des types de données à analyser
DATA_TYPES = ['real', 'simulated']

for data_type in DATA_TYPES:
    print(f"\n Début du traitement des données : {data_type.upper()}")
    
    TRAIN_DIR = CACHE_DIR / data_type / 'train'
    TEST_DIR = CACHE_DIR / data_type / 'test'
    VAL_DIR = CACHE_DIR / data_type / 'val'

    for split_dir in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        # Vérification de l'existence du dossier
        if not split_dir.exists():
            print(f"⚠️ Le dossier {split_dir} n'existe pas, passage au suivant.")
            continue

        split_name = split_dir.name.upper()
        type_label = "RÉEL" if data_type == 'real' else "SIMULÉ"
        suffix = "real" if data_type == 'real' else "sim"
        
        csv_files = list(split_dir.rglob("*_metrics.csv"))
        if len(csv_files) == 0:
            print(f"⚠️ Aucun fichier CSV trouvé dans {split_dir}, passage au suivant.")
            continue
            
        df_list = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # Sauvegarde du CSV global du split
        output_csv = split_dir / f"global_metrics_{split_dir.name}.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Fichier global créé : {output_csv}")

        # Config de la charte graphique
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(15, 14))

        # -------------------------------------------------------------------------
        # GRAPH 1 : Taux de Zéros dans le visible
        # -------------------------------------------------------------------------
        sns.histplot(df['zero_visible_percentile_hsi'] * 100, bins=200,binrange=(0,1),ax=axes[0, 0], color='crimson', kde=False)
        axes[0, 0].set_title(f"Taux de zéros dans le visible ({split_name}) {type_label}")
        axes[0, 0].set_xlabel("% de zéros (Bandes 5-35)")
        axes[0, 0].set_ylabel("Nombre de patchs")

        # -------------------------------------------------------------------------
        # GRAPH 2 : Couverture Nuageuse (Dynamic World)
        # -------------------------------------------------------------------------
        sns.histplot(df['percentile_of_cloud_dw'] * 100, bins=50, ax=axes[0, 1], color='skyblue', kde=False)
        axes[0, 1].set_title(f"Couverture nuageuse Dynamic World ({split_name}) {type_label}")
        axes[0, 1].set_xlabel("% de nuages du patch")
        axes[0, 1].set_ylabel("Nombre de patchs")
        
        # -------------------------------------------------------------------------
        # GRAPH 3 : Superposition des Réflectances Moyennes (HSI vs MSI)
        # -------------------------------------------------------------------------
        common_bins = 50
        common_range_mean = (0.0, 0.8)

        sns.histplot(df['mean_hsi'], bins=common_bins, binrange=common_range_mean, color='teal', label='Moyenne HSI', alpha=0.6, kde=False, ax=axes[1, 0])
        sns.histplot(df['mean_msi'], bins=common_bins, binrange=common_range_mean, color='orange', label='Moyenne MSI', alpha=0.6, kde=False, ax=axes[1, 0])
        axes[1, 0].set_title(f"Superposition des Réflectances Moyennes ({split_name}) {type_label}")
        axes[1, 0].set_xlabel("Valeur moyenne du patch")
        axes[1, 0].set_ylabel("Nombre de patchs")
        axes[1, 0].legend()
        
        # -------------------------------------------------------------------------
        # GRAPH 4 : Taux de pixels très brillants (Saturation)
        # -------------------------------------------------------------------------
        sns.histplot(df['brillant_visible_percentile_hsi'] * 100, bins=30, ax=axes[1, 1], color='gold', kde=False)
        axes[1, 1].set_title(f"Taux de pixels saturés / brillants ({split_name}) {type_label}")
        axes[1, 1].set_xlabel("% de pixels > 0.8 (Bandes 5-35)")
        axes[1, 1].set_ylabel("Nombre de patchs")

        # -------------------------------------------------------------------------
        # GRAPH 5 : Superposition des Écarts-Types (HSI vs MSI)
        # -------------------------------------------------------------------------
        common_range_std = (0.0, 0.3) 
        
        sns.histplot(df['std_hsi'], bins=common_bins, binrange=common_range_std, color='teal', label='Std HSI', alpha=0.6, kde=False, ax=axes[2, 0])
        sns.histplot(df['std_msi'], bins=common_bins, binrange=common_range_std, color='orange', label='Std MSI', alpha=0.6, kde=False, ax=axes[2, 0])
        axes[2, 0].set_title(f"Superposition des Écarts-Types ({split_name}) {type_label}")
        axes[2, 0].set_xlabel("Écart-type du patch")
        axes[2, 0].set_ylabel("Nombre de patchs")
        axes[2, 0].legend()

        # -------------------------------------------------------------------------
        # GRAPH 6 : Corrélation directe (Scatter Plot Moyennes)
        # -------------------------------------------------------------------------
        sns.scatterplot(data=df, x='mean_hsi', y='mean_msi', alpha=0.5, ax=axes[2, 1], color='purple')
        axes[2, 1].plot([0, 0.6], [0, 0.6], color='red', linestyle='--', label='y = x')
        axes[2, 1].set_title(f"Alignement des Moyennes ({split_name}) {type_label}")
        axes[2, 1].set_xlabel("Moyenne HSI")
        axes[2, 1].set_ylabel("Moyenne MSI")
        axes[2, 1].legend()

        # Finalisation et sauvegarde
        plt.tight_layout()
        output_png = split_dir / f"dataset_diagnostic_plots_{split_dir.name}_{suffix}.png"
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"📊 Graphique sauvegardé : {output_png}")

        cond_brillant = df['brillant_visible_percentile_hsi'] * 100 > 1.0
        
        # 2. Plus de 1% de pixels à zéro dans le visible
        cond_zeros = df['zero_visible_percentile_hsi'] * 100 > 0.1
        
        # 3. Présence de nuages (ex: supérieur à 10% ou présence stricte selon ton choix, ici > 10%)
        cond_nuages = df['percentile_of_cloud_dw'] * 100 > 10.0
        
        # Combinaison avec un OU (|)
        df_aberrant = df[cond_brillant | cond_zeros | cond_nuages]
        
        print(f"\n--- Split {split_dir.name} ({data_type}) ---")
        print(f"Total patchs analysés : {len(df)}")
        print(f"Nombre de patchs aberrants détectés : {len(df_aberrant)}")
        
        if len(df_aberrant) > 0:
            print("Liste des lignes / patchs à vérifier :")
            # Si tu as une colonne qui stocke le nom du fichier (ex: 'file_name' ou 'patch_id')
            colonne_nom = 'file_name' if 'file_name' in df.columns else df.columns[0]
            
            for idx, row in df_aberrant.iterrows():
                print(f"  - Index [Ligne {idx}] | ID/Nom: {row[colonne_nom]} | "
                      f"Sat: {row['brillant_visible_percentile_hsi']*100:.2f}% | "
                      f"Zeros: {row['zero_visible_percentile_hsi']*100:.2f}% | "
                      f"Nuages: {row['percentile_of_cloud_dw']*100:.2f}%")
        else:
            print(" Aucun patch aberrant sur ce split avec ces critères.")