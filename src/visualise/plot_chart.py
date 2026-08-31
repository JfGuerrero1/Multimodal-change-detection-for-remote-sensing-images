import matplotlib.pyplot as plt
import numpy as np

# Ordre des 5 modèles :
# 0: Simulé, 1: Réel, 2: Résiduel MSE, 3: Résiduel Classique, 4: Résiduel Normalisé
models = ['Simulé', 'Réel', 'Résiduel MSE', 'Résiduel Classique', 'Résiduel Normalisé']
metrics = ['PSNR', 'SSIM', 'SAM', 'ERGAS', 'MAE']
datasets = ['Test', 'Validation']

# Sens d'optimisation des métriques ('max' pour $\uparrow$, 'min' pour $\downarrow$)
optimization = {'PSNR': 'max', 'SSIM': 'max', 'SAM': 'min', 'ERGAS': 'min', 'MAE': 'min'}

# Moyennes (Means) pour les 5 modèles
means = {
    'PSNR': {
        'Test': [37.0396, 30.6194, 30.71, 31.3891, 32.4164], 
        'Validation': [36.7303, 29.8549, 29.36, 30.3270, 32.0000]
    },
    'SSIM': {
        'Test': [0.8637, 0.7760, 0.74, 0.7947, 0.7886], 
        'Validation': [0.9504, 0.8560, 0.79, 0.8206, 0.8371]
    },
    'SAM': {
        'Test': [0.1187, 0.1457, 0.16, 0.1404, 0.1253], 
        'Validation': [0.0611, 0.1109, 0.11, 0.1095, 0.0863]
    },
    'ERGAS': {
        'Test': [32.5804, 31.8856, 40.06, 28.8601, 23.2319], 
        'Validation': [18.3805, 20.5811, 54.96, 41.9610, 25.7896]
    },
    'MAE': {
        'Test': [0.0110, 0.0250, 0.0250, 0.0246, 0.0202], 
        'Validation': [0.0105, 0.0234, 0.0259, 0.0228, 0.0188]
    }
}

# Écarts-types (Standard Deviations) pour les 5 modèles
stds = {
    'PSNR': {
        'Test': [2.8357, 4.6423, 4.65, 6.0260, 5.1393], 
        'Validation': [3.2401, 3.5947, 2.80, 3.1043, 3.3570]
    },
    'SSIM': {
        'Test': [0.2222, 0.1945, 0.19, 0.1424, 0.1750], 
        'Validation': [0.0294, 0.0766, 0.09, 0.0801, 0.0720]
    },
    'SAM': {
        'Test': [0.1364, 0.1097, 0.13, 0.1051, 0.1061], 
        'Validation': [0.0285, 0.0775, 0.07, 0.0752, 0.0586]
    },
    'ERGAS': {
        'Test': [69.3824, 57.1047, 47.83, 23.3558, 34.2060], 
        'Validation': [18.0438, 18.3406, 34.29, 24.7703, 19.4503]
    },
    'MAE': {
        'Test': [0.0032, 0.0157, 0.0144, 0.0161, 0.0142], 
        'Validation': [0.0027, 0.0084, 0.0089, 0.0083, 0.0077]
    }
}

# Couleurs : Simulé=Rouge, Réel=Bleu, Résiduel MSE=Orange, Résiduel Classique=Vert foncé, Résiduel Normalisé=Vert clair
colors = ['red', 'blue', 'orange', 'forestgreen', 'limegreen']

x = np.arange(len(datasets))  # [0, 1] pour Test et Validation
width = 0.15                  # Largeur des barres pour 5 modèles

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    metric_means = means[metric]
    metric_stds = stds[metric]
    
    # Détermination du meilleur modèle (en excluant l'index 0 -> 'Simulé')
    best_indices = {}
    for ds_idx, ds in enumerate(datasets):
        # On extrait uniquement les indices de 1 à 4 (du Réel au Résiduel Normalisé)
        valid_indices = range(1, len(models))
        vals = [metric_means[ds][m_idx] for m_idx in valid_indices]
        
        if optimization[metric] == 'max':
            best_local_idx = np.argmax(vals)
        else:
            best_local_idx = np.argmin(vals)
            
        best_indices[ds] = valid_indices[best_local_idx]

    # Tracer les barres pour chaque modèle
    for m_idx, model_name in enumerate(models):
        offset = (m_idx - 2) * width  # Centrage des 5 barres
        
        for i, ds in enumerate(datasets):
            val = metric_means[ds][m_idx]
            err = metric_stds[ds][m_idx]
            
            # Le modèle simulé (m_idx == 0) ne reçoit jamais les hachures
            is_best = (m_idx == best_indices[ds] and m_idx != 0)
            
            ax.bar(x[i] + offset, val, width,
                   yerr=err,
                   color=colors[m_idx],
                   edgecolor='black',
                   hatch='//' if is_best else '',
                   capsize=2,
                   label=model_name if i == 0 and idx == 0 else "")

    ax.set_title(f'Métrique : {metric}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Supprimer le 6ème subplot vide s'il y en a un
if len(metrics) < len(axes):
    fig.delaxes(axes[-1])

# Légende globale en haut
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("comparison_metrics_5models_no_simulated_best.png", dpi=300)
plt.show()

