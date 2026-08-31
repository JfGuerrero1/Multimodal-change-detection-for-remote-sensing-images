import os
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
PROJECT_PATH = "JfGuerrero/Multimodal-change-detection"

RUN_IDS = [
    "whs69rcw",
    "5s7c0uvn",
    "6cerivdu",
    "8p13wa1e",
    "01bdn81w",
    "9knzit7w",
    "5dmh1fmi",
    "z4dp4z5s",
    "r3pubh42",
]

RUN_NAMES_MAPPING = {
    "whs69rcw": "Modèle résiduel MSE",
    "5s7c0uvn": "Modèle résiduel normalisé",
    "6cerivdu": "Modèle résiduel",
    "8p13wa1e": "Modèle simulé",
    "01bdn81w": "Modèle résiduel normalisé",
    "9knzit7w": "Modèle filtered after",
    "5dmh1fmi": "Modèle U-Net classique",
    "z4dp4z5s": "Modèle all wave after",
    "r3pubh42": "Modèle all wave",
}

OUTPUT_CSV = "runs_metrics_filtered.csv"
CLEANED_CSV = "runs_metrics_cleaned.csv"
PLOTS_DIR = "tournois_plots"

TARGET_METRICS = [
    "lr",
    "val_ssim",
    "val_psnr",
    "val_rmse",
    "val_mae",
    "val_sam",
    "val_ergas",
    "val_loss",
    "train_loss",
    "train_mae",
    "train_sam",
    "train_grad_norm",
    "train_ssim",
]

run_labels = {
    "whs69rcw": "Modèle résiduel MSE",
    "5s7c0uvn": "Modèle résiduel normalisé",
    "6cerivdu": "Modèle résiduel",
    "8p13wa1e": "Modèle simulé",
    "01bdn81w": "Modèle résiduel normalisé (2)",
    "9knzit7w": "Modèle filtered after",
    "r3pubh42": "Modèle all wave",
    "z4dp4z5s": "Modèle all wave after",
    "5dmh1fmi": "Modèle U-Net classique",
}

tournois = {
    "1. Résiduel vs Filtered After vs All Wave vs All Wave After": [
        "6cerivdu",
        "9knzit7w",
        "r3pubh42",
        "z4dp4z5s",
    ],
    "2. U-Net Classique vs Résiduel": ["5dmh1fmi", "6cerivdu"],
    "3. U-Net Classique vs Simulé": ["5dmh1fmi", "8p13wa1e"],
    "4. Résiduel MSE vs Résiduel ": ["whs69rcw", "6cerivdu"],
    "5. Résiduel vs Résiduel Normalisé": ["6cerivdu", "5s7c0uvn"],
}

# ERGAS a été retiré ici
toutes_les_metriques = [
    "loss",
    "mae",
    "ssim",
    "psnr",
    "rmse",
    "sam",
    "grad_norm",
]


# ==============================================================================
# 2. EXTRACTION WANDB
# ==============================================================================
def export_filtered_metrics(
    project_path, run_ids, output_filename, metrics_to_keep, name_mapping
):
  api = wandb.Api()
  all_dfs = []

  print(f"Début du rapatriement et filtrage pour {len(run_ids)} run(s)...")

  for idx, run_id in enumerate(run_ids, 1):
    try:
      full_run_path = f"{project_path}/{run_id}"
      run = api.run(full_run_path)
      custom_name = name_mapping.get(run_id, run.name)
      print(f"[{idx}/{len(run_ids)}] Récupération : {custom_name} ({run_id})...")

      df = run.history(samples=10000)
      if df.empty:
        continue

      base_cols = [col for col in ["_step", "epoch"] if col in df.columns]
      available_metrics = [m for m in metrics_to_keep if m in df.columns]
      df_filtered = df[base_cols + available_metrics].copy()

      df_filtered.insert(0, "run_name", custom_name)
      df_filtered.insert(0, "run_id", run.id)
      all_dfs.append(df_filtered)
    except Exception as e:
      print(f"  ❌ Erreur pour {run_id} : {e}")

  if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_filename, index=False)
    print(f"✅ Export réussi : {output_filename}")


# ==============================================================================
# 3. FONCTION DE SAUVEGARDE (SÉPARATION PROPRE TRAIN / VAL)
# ==============================================================================
def plot_and_save_grouped_runs(
    csv_path, runs_list, labels_list, tournament_name, metrics, output_dir
):
  df = pd.read_csv(csv_path)
  x_axis = "epoch" if "epoch" in df.columns else "_step"

  if labels_list is None:
    labels_list = runs_list

  os.makedirs(output_dir, exist_ok=True)

  safe_tournament_name = "".join(
      c if c.isalnum() or c in (" ", "_", "-") else "_" for c in tournament_name
  ).strip()
  safe_tournament_name = safe_tournament_name.replace(" ", "_")
  tourney_folder = os.path.join(output_dir, safe_tournament_name)
  os.makedirs(tourney_folder, exist_ok=True)

  # --- 1. GRAPHIQUE TRAIN ---
  train_metrics_available = [
      m for m in metrics if f"train_{m}" in df.columns
  ]

  if train_metrics_available:
    ncols = 2 if len(train_metrics_available) > 1 else 1
    nrows = (len(train_metrics_available) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 7, nrows * 4), sharex=True, squeeze=False
    )
    axes = axes.flatten()

    for i, metric in enumerate(train_metrics_available):
      ax = axes[i]
      train_col = f"train_{metric}"

      for run_target, custom_label in zip(runs_list, labels_list):
        group = df[
            (df["run_id"] == run_target) | (df["run_name"] == run_target)
        ]
        if group.empty:
          continue
        group = group.sort_values(by=x_axis)
        ax.plot(
            group[x_axis], group[train_col], linewidth=1.5, label=custom_label
        )

      ax.set_title(f"Train - {metric.upper()}", fontsize=11, fontweight="bold")
      ax.set_xlabel("Époque / Step", fontsize=9)
      ax.set_ylabel(metric.capitalize(), fontsize=9)
      ax.grid(True, linestyle="--", alpha=0.5)
      ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

    plt.suptitle(
        f"{tournament_name} — Toutes les Métriques TRAIN",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(tourney_folder, "all_metrics_TRAIN.png"), dpi=300)
    plt.close()

  # --- 2. GRAPHIQUE VALIDATION ---
  val_metrics_available = [m for m in metrics if f"val_{m}" in df.columns]

  if val_metrics_available:
    ncols = 2 if len(val_metrics_available) > 1 else 1
    nrows = (len(val_metrics_available) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 7, nrows * 4), sharex=True, squeeze=False
    )
    axes = axes.flatten()

    for i, metric in enumerate(val_metrics_available):
      ax = axes[i]
      val_col = f"val_{metric}"

      for run_target, custom_label in zip(runs_list, labels_list):
        group = df[
            (df["run_id"] == run_target) | (df["run_name"] == run_target)
        ]
        if group.empty:
          continue
        group = group.sort_values(by=x_axis)
        ax.plot(
            group[x_axis], group[val_col], linewidth=1.5, label=custom_label
        )

      ax.set_title(
          f"Validation - {metric.upper()}", fontsize=11, fontweight="bold"
      )
      ax.set_xlabel("Époque / Step", fontsize=9)
      ax.set_ylabel(metric.capitalize(), fontsize=9)
      ax.grid(True, linestyle="--", alpha=0.5)
      ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

    plt.suptitle(
        f"{tournament_name} — Toutes les Métriques VALIDATION",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(tourney_folder, "all_metrics_VAL.png"), dpi=300)
    plt.close()


# ==============================================================================
# 4. EXÉCUTION PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
  # Décommente si tu veux relancer le téléchargement complet depuis Wandb :
  # export_filtered_metrics(PROJECT_PATH, RUN_IDS, OUTPUT_CSV, TARGET_METRICS, RUN_NAMES_MAPPING)

  try:
    df = pd.read_csv(OUTPUT_CSV)
    df_epochs = df.dropna(subset=["epoch"]).copy()
    df_epochs["epoch"] = df_epochs["epoch"].astype(int)
    df_epochs.to_csv(CLEANED_CSV, index=False)
    print(f"✅ Fichier nettoyé : {CLEANED_CSV} ({len(df_epochs)} lignes)")
  except Exception as e:
    print(f"⚠️ Erreur lors du nettoyage : {e}")

  print(
      f"\n🏆 Génération des images groupées (sans markers) dans"
      f" '{PLOTS_DIR}/'..."
  )
  for titre_tournoi, ids_groupe in tournois.items():
    print(f"➡️ Traitement : {titre_tournoi}")
    labels_groupe = [run_labels[rid] for rid in ids_groupe]

    plot_and_save_grouped_runs(
        csv_path=CLEANED_CSV,
        runs_list=ids_groupe,
        labels_list=labels_groupe,
        tournament_name=titre_tournoi,
        metrics=toutes_les_metriques,
        output_dir=PLOTS_DIR,
    )

  print("\n✨ Terminé !")