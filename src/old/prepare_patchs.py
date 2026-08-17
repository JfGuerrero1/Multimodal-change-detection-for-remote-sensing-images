from old.utils_dataset import *
import argparse
import glob

TEST_SCENES= ["baltijsk", "camerino", "codigoro", "copenhagen", "cullivel", "jagersfontein", "kirtland", "lorca"]
VAL_SCENES = ["arborea", "athens", "beer_sheva", "istanbul", "los_cabos", "taiwan", "yuen_long"]
TRAIN_SCENES= [
    "aranjuez","bari", "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "cape_town", "copperton",
    "cukotka", "dellys", "dubai", "dublin", "elsalto", "eyjafjoll", "fontainebleau", "fukushima",
    "guantanamo", "hanging_rock", "java", "jordan", "kitami", "lagos", "london", "los_angeles",
    "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
    "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
    "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
    "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"
]

def genere_interp(mode,split):
    split_cache_dir = CACHE_DIR / mode / split       
    print(f"Traitement de {mode}/{split}...")
        
    for x_path in glob.glob(str(split_cache_dir / "*_X_patches.npy")):
        interp_path = x_path.replace('_X_patches.npy', '_interp_patches.npy')
        if Path(interp_path).exists():
            continue
                
        x_mmap = np.load(x_path, mmap_mode='r')
        n, c, h, w = x_mmap.shape
        c_hsi = INTERP_MATRIX.shape[0]
            
        interp_mmap = np.lib.format.open_memmap(
            interp_path, mode='w+', dtype=np.float32, 
            shape=(n, c_hsi, h, w)
            )
        for i in tqdm(range(n), desc=Path(x_path).name):
            interp_numpy = INTERP_MATRIX @ x_mmap[i].reshape(c, -1)
            interp_mmap[i] = interp_numpy.reshape(c_hsi, h, w)
            
        interp_mmap.flush()
        del interp_mmap
        print(f"  -> {interp_path} créé")
    

def prepare_dataset_offline(scene_pairs, split_name, patch_size=256, use_simulated_msi=False):


    mode_suffix = "simulated" if use_simulated_msi else "real"
    split_cache_dir = CACHE_DIR / mode_suffix / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    sim_str = 'sim' if use_simulated_msi else 'no_sim'
    
    if use_simulated_msi:
        c_multi = SRF_MATRIX.shape[1] #12
    
    c_hsi = INTERP_MATRIX.shape[0]  # 230

    all_stats_msi=[]
    all_stats_hsi=[]
    for x_file, y_file in tqdm(scene_pairs, desc=f"Patching {split_name}"):
        base_name = y_file.name.replace('-prs.nc', '')
        scene_name=base_name.replace('-after','')
        print(scene_name)
        print(base_name)
        print(DATA_DIR/scene_name)
        
        x_path = split_cache_dir / f'{base_name}_X_patches.npy'
        y_path = split_cache_dir / f'{base_name}_y_patches.npy'
        dw_path=DATA_DIR/scene_name/f'{base_name}-dw.nc'
        interp_path = split_cache_dir / f'{base_name}_interp_patches.npy'  

        #if x_path.exists() and y_path.exists() and interp_path.exists():  
            #continue

            
        with xr.open_dataset(x_file) as ds_x, xr.open_dataset(y_file) as ds_y,xr.open_dataset(dw_path)as ds_dw:
            h, w, c_hyper = ds_y["sr"].shape
            map=ds_dw["lcc"]
            print(map.shape)
            if not use_simulated_msi:
                c_multi = ds_x["sr"].shape[2] #12
            
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)
            total_patches = (h_crop // patch_size) * (w_crop // patch_size)
            
            X_mmap = np.lib.format.open_memmap(str(x_path), mode='w+', dtype=np.float32, shape=(total_patches, c_multi, patch_size, patch_size))
            y_mmap = np.lib.format.open_memmap(str(y_path), mode='w+', dtype=np.float32, shape=(total_patches, c_hyper, patch_size, patch_size))
            interp_mmap = np.lib.format.open_memmap(str(interp_path), mode='w+', dtype=np.float32, shape=(total_patches, c_hsi, patch_size, patch_size))

            scene_preview_dir = split_cache_dir / f"{base_name}_previews"
            scene_preview_dir.mkdir(parents=True, exist_ok=True)  
            
            patch_idx = 0
            for i in range(h_crop // patch_size):
                for j in range(w_crop // patch_size):
                    r = i * patch_size
                    c = j * patch_size
                    
                    patch_hyper = ds_y["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32) #h,w,c
                    patch_dw=map[r:r+patch_size,c:c+patch_size]

                    if use_simulated_msi:
                        hyper_2d = patch_hyper.reshape(-1, c_hyper) #h*w,c_hsi
                        patch_multi = np.dot(hyper_2d, SRF_MATRIX).reshape(patch_size, patch_size, c_multi).astype(np.float32)
                        #                    h*w,c_hsi * c_hsi,c_msi
                        #finally, h,w,c_multi
                    else:
                        patch_multi = ds_x["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32)
                    
                    patch_multi_chw = np.transpose(patch_multi, (2, 0, 1))  # [C_msi, H, W]
                    X_mmap[patch_idx] = patch_multi_chw
                    patch_hyper_chw= np.transpose(patch_hyper, (2, 0, 1)) # [C_hsi,H,W]
                    y_mmap[patch_idx] = patch_hyper_chw
              
                    interp_numpy = INTERP_MATRIX @ patch_multi_chw.reshape(c_multi, -1)  # [C_hsi, H*W]
                    #              C_HSI,C_MSI @ C_MSI, H*W
                
                    
                    interp_mmap[patch_idx] = interp_numpy.reshape(c_hsi, patch_size, patch_size) #[C_HSI_,H,W]

                   

                    #affichage, MSI,HSI,DW
                    
                    name_id = f"{base_name}_patch_{patch_idx:04d}_{sim_str}"

                    #tryptique_view(patch_multi_chw,patch_hyper_chw,patch_dw,scene_preview_dir/f'{name_id}_tryptique.png')

                    #fichier json
                    metrics = get_stat_from_patch(patch_multi_chw, name_id)
                    all_stats_msi.append(metrics)
                    metrics = get_stat_from_patch(patch_hyper_chw, name_id)
                    all_stats_hsi.append(metrics)

                    stat_csv(patch_hyper_chw,patch_multi_chw,patch_dw,name_id,scene_preview_dir)



                    patch_idx += 1


                
def create_data_loaders_spectral( use_simulated_msi, augment, batch_size=8,  num_workers=4, is_residual=False,keep_atm_wave=True,is_normalised=False):
    

    print(f" Chargement direct des patches pré-calculés depuis : {CACHE_DIR.resolve()}")
    
    mode_suffix = "simulated" if use_simulated_msi else "real"
    
    # Chemins directs vers tes dossiers déjà splittés
    train_dir = CACHE_DIR / mode_suffix / 'train'
    val_dir = CACHE_DIR / mode_suffix / 'val'
    test_dir = CACHE_DIR / mode_suffix / 'test'
    
    # Instanciation directe des datasets sans refaire de découpage
    train_dataset = SpectralDataset(train_dir, augment=augment, is_residual=is_residual,keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)
    val_dataset = SpectralDataset(val_dir, augment=False, is_residual=is_residual, keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)
    test_dataset = SpectralDataset(test_dir, augment=False, is_residual=is_residual,keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset by creating patches.")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Name of the split")
    parser.add_argument("--simulated", action="store_true", dest="use_simulated_msi",
                        help="Enable MSI simulation with SRF response matrix")
    
    args = parser.parse_args()
    PATCH_SIZE = args.patch_size
    SPLIT_NAME = args.split
    USE_SIMULATED_MSI = args.use_simulated_msi
    y_files = sorted(glob.glob(str(DATA_DIR / "**" / "*after-prs.nc")))
    
    scene_pairs = []
    for y_file in y_files:
        y_path = Path(y_file)
        scene_name = y_path.parent.name

        if SPLIT_NAME == "train" and scene_name not in TRAIN_SCENES:
            continue
        elif SPLIT_NAME == "val" and scene_name not in VAL_SCENES:
            continue
        elif SPLIT_NAME == "test" and scene_name not in TEST_SCENES:
            continue
        
        if USE_SIMULATED_MSI:
            x_path = y_path 
        else:
            x_path = y_path.parent / y_path.name.replace('-prs.nc', '-s2.nc') 
            
            if not x_path.exists():
                print(f"⚠️ Attention : Fichier MSI réel introuvable pour {y_path.name}, skipping.")
                continue
                
        scene_pairs.append((x_path, y_path))

    prepare_dataset_offline(
        scene_pairs=scene_pairs, 
        split_name=SPLIT_NAME, 
        patch_size=PATCH_SIZE, 
        use_simulated_msi=USE_SIMULATED_MSI
    )

if __name__ == "__main__":
