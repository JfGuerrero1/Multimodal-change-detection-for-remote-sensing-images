import matplotlib.pyplot as plt
import numpy as np
import pathlib

import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import WVL_PRS,SRF_MATRIX


# Supposons que :
# - 'srf_matrix' est votre matrice (de dimensions : nb_bandes_hsi x nb_bandes_msi)
# - 'wavelengths' est un tableau contenant la longueur d'onde de chaque bande HSI (ex: en nanomètres)

plt.figure(figsize=(10, 6))
wavelengths = WVL_PRS  # Longueurs d'onde HSI (en nm)
srf_matrix = SRF_MATRIX  # Matrice de réponse spectrale (SRF) du capteur MSI
# On trace chaque colonne (qui correspond à une bande MSI) en fonction des longueurs d'onde HSI
plt.plot(wavelengths, srf_matrix)

plt.title("Fonctions de Réponse Spectrale (SRF) du capteur MSI")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Sensibilité ")
plt.grid(True)
plt.savefig("Srf")