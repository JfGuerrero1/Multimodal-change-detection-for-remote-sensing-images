import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ==========================================
# CONSTANTES ET CONFIGURATION PRISMA
# ==========================================
WVL_PRS = np.array([
    406.9934, 415.839 , 423.78476, 431.3347 , 438.6569 , 446.0147 , 453.38947, 460.73175, 468.09842, 475.31885,
    482.54816, 489.79486, 497.05865, 504.51172, 512.0464 , 519.54376, 527.3053 , 535.05255, 542.88513, 550.9146 ,
    559.02026, 567.2061 , 575.4868 , 583.8441 , 592.339  , 601.0144 , 609.9582 , 618.72   , 627.77844, 636.6763 ,
    645.9638 , 655.41876, 664.8941 , 674.46436, 684.13727, 694.12836, 703.737  , 713.72687, 723.87994, 733.9552 ,
    744.14954, 754.4696 , 764.85645, 775.2735 , 785.65955, 796.127  , 806.71106, 817.31104, 827.9195 , 838.5272 ,
    849.20996, 859.97314, 870.74255, 881.45605, 892.08093, 902.80164, 913.44507, 923.9502 , 934.11206, 944.6273 ,
    956.2715 , 967.0267 , 977.3654 , 979.224  , 988.9179 , 998.9082 , 1008.6443, 1018.5357, 1029.344  , 1037.9878,
    1047.675 , 1057.5737, 1067.7948, 1078.2161, 1088.761 , 1099.2776, 1109.8894, 1120.6759, 1131.3048, 1142.0703,
    1152.6501, 1163.676 , 1174.7142, 1185.5884, 1196.3394, 1207.2737, 1217.8635, 1229.1852, 1240.2145, 1250.9799,
    1262.5322, 1273.4963, 1284.4878, 1295.4218, 1306.218 , 1317.2566, 1328.2993, 1339.1294, 1349.7877, 1361.0531,
    1372.9117, 1383.2798, 1394.754 , 1405.6268, 1416.5374, 1427.3748, 1438.466 , 1449.1888, 1459.3157, 1469.9308,
    1480.8422, 1491.4292, 1502.0236, 1512.6333, 1523.2222, 1533.7764, 1544.2262, 1554.8168, 1565.3688, 1575.6274,
    1585.8597, 1596.2454, 1606.4913, 1616.8336, 1627.021 , 1637.0919, 1647.2316, 1656.933 , 1667.185 , 1677.3193,
    1687.4269, 1697.2943, 1707.0945, 1716.8589, 1726.6516, 1736.4883, 1746.2192, 1755.833 , 1765.5127, 1775.1178,
    1784.7173, 1793.9531, 1803.5902, 1813.0514, 1822.4413, 1832.0272, 1841.3256, 1850.5543, 1859.5587, 1868.1732,
    1878.7426, 1887.081 , 1896.0913, 1904.9347, 1914.3015, 1923.3857, 1932.2599, 1941.1107, 1949.9008, 1958.6244,
    1967.3418, 1976.013 , 1984.853 , 1993.5482, 2002.1106, 2010.6614, 2019.3214, 2027.7267, 2036.2607, 2044.6809,
    2053.0078, 2061.3787, 2069.7957, 2077.9915, 2086.3823, 2094.6252, 2102.8213, 2111.039 , 2119.2314, 2127.3372,
    2135.5103, 2143.4656, 2151.3862, 2159.564 , 2167.4849, 2175.3442, 2183.4202, 2191.1003, 2199.1353, 2206.843 ,
    2214.625 , 2222.4263, 2230.0076, 2237.904 , 2245.4485, 2253.1104, 2260.8665, 2268.2883, 2276.0537, 2283.4934,
    2290.8267, 2298.6094, 2305.7227, 2313.2007, 2320.8955, 2327.8242, 2335.5264, 2342.8228, 2349.7915, 2357.2937,
    2364.5945, 2371.5522, 2378.771 , 2386.0618, 2393.0388, 2400.036 , 2407.6045, 2414.3567, 2421.2373, 2428.6677,
    2435.5442, 2442.403 , 2449.1423, 2456.5857, 2463.0303, 2469.6272, 2477.055 , 2483.793 , 2490.2192, 2497.1155
])

def find_index(target_nm):
    return np.argmin(np.abs(WVL_PRS - target_nm))

prisma_clean = [489.79, 550.91, 664.89, 1606.49, 2214.62] 
prisma_atm   = [1383.27, 1416.53, 1896.09, 1914.30, 1941.11]  # Nouvelle bande atmosphérique intégrée

good_indices = [find_index(t) for t in prisma_clean]
bad_indices  = [find_index(t) for t in prisma_atm]

def robust_scale(data):
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return data
    p2, p98 = np.percentile(valid, (2, 98))
    if p98 == p2:
        return data
    return np.clip((data - p2) / (p98 - p2), 0, 1)

# ==========================================
# 1. TRAITEMENT PRISMA
# ==========================================
def process_prisma(nc_path, variable_name='sr'):
    ds = xr.open_dataset(nc_path)
    cube = ds[variable_name]
    
    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    fig.suptitle('PRISMA : Vue RGB et Sélection de Bandes Spectrales', fontsize=14, fontweight='bold', y=0.96)
    
    # RGB sur la première colonne (partagé sur les 2 lignes)
    r_idx = find_index(664.89)
    g_idx = find_index(550.91)
    b_idx = find_index(489.79)
    rgb_stack = np.dstack([
        robust_scale(cube.isel(cw=r_idx).values),
        robust_scale(cube.isel(cw=g_idx).values),
        robust_scale(cube.isel(cw=b_idx).values)
    ])
    
    for row in range(2):
        axes[row, 0].imshow(rgb_stack)
        axes[row, 0].set_title("Composition RGB\nNaturelle", fontsize=10, fontweight='bold', pad=6,
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        axes[row, 0].axis('off')

    # Ligne du haut : Bandes de surface
    for i, idx in enumerate(good_indices):
        wvl = WVL_PRS[idx]
        cw_data = cube.isel(cw=idx).values
        axes[0, i+1].imshow(robust_scale(cw_data), cmap='gray')
        axes[0, i+1].set_title(f"{wvl:.1f} nm", fontsize=10, fontweight='bold', pad=6,
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        axes[0, i+1].axis('off')
        
    # Ligne du bas : Bandes atmosphériques
    for i, idx in enumerate(bad_indices):
        wvl = WVL_PRS[idx]
        cw_data = cube.isel(cw=idx).values
        axes[1, i+1].imshow(robust_scale(cw_data), cmap='gray')
        axes[1, i+1].set_title(f"{wvl:.1f} nm", fontsize=10, fontweight='bold', pad=6,
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        axes[1, i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig('prisma_planche_globale.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 10 spectres aléatoires
    shape = cube.shape
    cw_axis = np.argmin(np.abs(np.array(shape) - 230))
    spatial_axes = [i for i in range(3) if i != cw_axis]
    h_dim, w_dim = shape[spatial_axes[0]], shape[spatial_axes[1]]
    
    np.random.seed(42)
    rand_y = np.random.randint(0, h_dim, 10)
    rand_x = np.random.randint(0, w_dim, 10)
    
    plt.figure(figsize=(10, 5))
    for idx_p, (y, x) in enumerate(zip(rand_y, rand_x)):
        if cw_axis == 2:
            spectrum = cube.isel({cube.dims[spatial_axes[0]]: y, cube.dims[spatial_axes[1]]: x}).values
        else:
            spectrum = cube.values[:, y, x] if cw_axis == 0 else cube.values[y, x, :]
            
        if len(spectrum) != len(WVL_PRS):
            spectrum = cube.values[y, x, :] if cw_axis == 2 else cube.values[:, y, x]

        plt.plot(WVL_PRS, spectrum, alpha=0.7, lw=1.2, label=f'Pixel {idx_p+1}')

    plt.title('Spectres PRISMA de 10 pixels aléatoires (400 - 2500 nm)', fontsize=12, fontweight='bold')
    plt.xlabel('Longueur d\'onde (nm)', fontsize=10)
    plt.ylabel('Réflectance de surface', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig('prisma_10_spectres.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. TRAITEMENT SENTINEL-2 (Avec longueurs d'onde)
# ==========================================
def process_sentinel2(nc_path, variable_name='sr'):
    ds = xr.open_dataset(nc_path)
    cube = ds[variable_name]
    
    fig, axes = plt.subplots(1, 6, figsize=(22, 4.5))
    fig.suptitle('Sentinel-2 : Vue RGB et Comparaison des Résolutions Spatiales', fontsize=14, fontweight='bold', y=0.98)
    
    try:
        r_s2 = cube.isel(band=3).values
        g_s2 = cube.isel(band=2).values
        b_s2 = cube.isel(band=1).values
    except:
        r_s2 = cube.isel(cw=3).values
        g_s2 = cube.isel(cw=2).values
        b_s2 = cube.isel(cw=1).values
        
    rgb_s2 = np.dstack([robust_scale(r_s2), robust_scale(g_s2), robust_scale(b_s2)])
    
    axes[0].imshow(rgb_s2)
    axes[0].set_title("RGB (10 m)\n(B4, B3, B2)", fontsize=10, fontweight='bold', pad=6,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    axes[0].axis('off')

    # Bandes Sentinel-2 avec leurs résolutions ET longueurs d'onde centrales approximatives
    s2_bands = [
        {"name": "B1 (Côtière)", "wvl": "443 nm", "res": "60 m", "idx": 0},
        {"name": "B4 (Rouge)", "wvl": "665 nm", "res": "10 m", "idx": 3},
        {"name": "B8 (NIR)", "wvl": "842 nm", "res": "10 m", "idx": 7},
        {"name": "B11 (SWIR 1)", "wvl": "1610 nm", "res": "20 m", "idx": 8},
        {"name": "B12 (SWIR 2)", "wvl": "2190 nm", "res": "20 m", "idx": 9}
    ]
    
    for i, b in enumerate(s2_bands):
        try:
            band_data = cube.isel(band=b["idx"]).values
        except:
            band_data = cube.isel(cw=b["idx"]).values
            
        axes[i+1].imshow(robust_scale(band_data), cmap='gray')
        axes[i+1].set_title(f"{b['name']} ({b['wvl']})\n{b['res']}", fontsize=10, fontweight='bold', pad=6,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig('sentinel2_planche_globale.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# EXÉCUTION GLOBALE
# ==========================================
prisma_nc_file = '/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/brasilia/brasilia-after-prs.nc'
s2_nc_file =     '/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/brasilia/brasilia-after-s2.nc'

process_prisma(prisma_nc_file, variable_name='sr')
process_sentinel2(s2_nc_file, variable_name='sr')
print("Exécution terminée avec succès !")