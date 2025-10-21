import glob
import os
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import Point
from libpysal import weights

# Machine Learning libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ConstrainedAgglomerativeClustering module contains modified version of agglomerative clustring algorithum. Make sure 'agg_clustring_final.py' is accessible.
from agg_clustring_final import ConstrainedAgglomerativeClustering


# Input files
NC_FILES_PATTERN = '../perception_module/output_data/*.nc'
GSV_METADATA_PATH = '../svi_module/svi_info.csv'
ET_CELLS_PATH = '../raw_data/ET_Cells_Glasgow/et_cells_glasgow_epsg27700.gpkg'

# Output files
OUTPUT_GPKG_PATH = './output_data/et_cells_glasgow_epsg27700_score_custer_all.gpkg'

# --- Functions ---
def process_embeddings(nc_file_path: str, variance_threshold: int = 10) -> pd.DataFrame:
    """
    Loads embeddings from a NetCDF file, applies PCA, extracts perception scores,
    and returns a DataFrame with PCA components and scores.

    Args:
        nc_file_path (str): Path to the NetCDF file containing embeddings.
        variance_threshold (int): Number of PCA components to retain.

    Returns:
        pd.DataFrame: DataFrame containing image IDs, PCA components, and perception scores.
    """
    print(f"Loading embeddings from {nc_file_path}")
    da_embs = xr.open_dataset(nc_file_path)

    print(f"Applying PCA (retaining {variance_threshold} components)")
    pca = PCA(n_components=variance_threshold)
    arr_pca = pca.fit_transform(da_embs.embeddings)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    perception_name_raw = str(da_embs.perception_type.item()) # .item() to get scalar value from xarray DataArray
    perception_name_cleaned = perception_name_raw.split('_')[-1] if '_' in perception_name_raw else perception_name_raw

    score = da_embs[f'{perception_name_raw}_score'].values

    # Create DataFrame with PCA components
    column_names = [f'pc_{perception_name_cleaned}_{i+1}' for i in range(arr_pca.shape[1])]
    pca_df = pd.DataFrame(data=arr_pca, columns=column_names)
    pca_df[f'score_{perception_name_cleaned}'] = score
    pca_df['image_id'] = da_embs.image.values # .values to get numpy array from xarray DataArray

    print(f"{perception_name_cleaned} processed {len(pca_df)} images with {len(column_names)} PCA components")
    return pca_df


# --- Main Script ---

if __name__ == "__main__":
    # 1. Load Data
    print("\n--- Loading Input Data ---")
    nc_files = glob.glob(NC_FILES_PATTERN)
    gsv_metadata = pd.read_csv(GSV_METADATA_PATH)
    et_cells = gpd.read_file(ET_CELLS_PATH)

    print(f"Found {len(nc_files)} NetCDF embedding files.")
    print(f"Loaded GSV metadata with {len(gsv_metadata)} entries.")
    print(f"Loaded ET cells GeoDataFrame with {len(et_cells)} entries.")

    # 2. Process Embeddings
    print("\n--- Processing Embeddings ---")
    pca_list = []
    for nc_file_path in nc_files:
        print(f"\nProcessing file: {os.path.basename(nc_file_path)}")
        pca_list.append(process_embeddings(nc_file_path))


    pca_lists_indexed = [df.set_index('image_id') for df in pca_list]
    pca_df = pd.concat(pca_lists_indexed, axis=1).reset_index()

    # 3. Merge with Metadata and Create GeoDataFrame
    print("\n--- Merging Data and Creating GeoDataFrame ---")
    pca_df = pd.merge(pca_df, gsv_metadata, left_on='image_id', right_on='pano_id', how='inner')
    pca_df.drop(columns=['lon', 'lat', 'pano_id', 'pano_heading', 'pano_pitch', 'pano_roll', 'status'], axis=1, inplace=True)
    pca_df = pca_df.astype({"pano_lon": float, "pano_lat": float})

    geometry = [Point(xy) for xy in zip(pca_df.pano_lon, pca_df.pano_lat)]
    pca_gdf = gpd.GeoDataFrame(pca_df, crs="EPSG:4326", geometry=geometry)
    pca_gdf_27700 = pca_gdf.to_crs('epsg:27700')

    print(f"Merged PCA data with metadata. Resulting GeoDataFrame has {len(pca_gdf_27700)} entries.")

    # 4. Spatial Join with ET Cells
    print("\n--- Performing Spatial Join ---")
    # Using sjoin_nearest to find the closest image to each ET cell
    et_cells_data = gpd.sjoin_nearest(et_cells, pca_gdf_27700, how='left', distance_col='dist')

    # Clean joined data: keep only the closest image for each hindex
    et_cells_clean_data = et_cells_data.sort_values('dist').drop_duplicates('hindex').sort_index()

    print(f"ET cells after spatial join and cleaning: {len(et_cells_clean_data)} entries.")

    # 5. Prepare Connectivity Weights for Clustering
    print("\n--- Preparing Connectivity Weights ---")
    # Using KNN weights for spatial constraint
    # Ensure there are enough points for KNN, otherwise KNN.from_dataframe might fail
    if len(et_cells_clean_data) > 5:
        w = weights.distance.KNN.from_dataframe(et_cells_clean_data, k=5)
        print(f"Created KNN spatial weights matrix with k=5 for {w.n} observations.")
    else:
        print("Not enough data points for KNN weights (less than 5). Clustering might proceed without connectivity.")
        w = None # Or create a trivial weights object if ConstrainedAgglomerativeClustering expects it

    # 6. Perform Constrained Agglomerative Clustering for each Perception Parameter
    print("\n--- Performing Constrained Agglomerative Clustering ---")
    perception_columns = [col for col in et_cells_clean_data.columns if col.startswith('pc_')]
    perceptions = set([col.split('_')[-2] for col in perception_columns])

    if not perceptions:
        print("No perception parameters found for clustering. Skipping clustering step.")
    else:
        for perception in perceptions:
            print(f"\nProcessing perception parameter: {perception}")

            # Extract relevant embedding columns
            X_embeddings = et_cells_clean_data[[col for col in et_cells_clean_data.columns if col.startswith(f'pc_{perception}')]].values
            
            if X_embeddings.shape[0] == 0:
                print(f"No data for perception '{perception}'. Skipping.")
                continue

            # Scale embeddings
            scaler = MinMaxScaler()
            emb_distances = scaler.fit_transform(X_embeddings)

            # Perform constrained clustering
            if w and w.sparse is not None:
                connectivity_matrix = w.sparse
            else:
                connectivity_matrix = None # No connectivity if w is None or sparse is not available

            try:
                clustering = ConstrainedAgglomerativeClustering(
                    n_clusters=30,
                    min_cluster_size=800,
                    connectivity=connectivity_matrix,
                    n_jobs=-1
                )
                labels = clustering.fit_predict(emb_distances)
                et_cells_clean_data[f'{perception}_ID'] = clustering.labels_
                print(f"Clustering complete for '{perception}'. Found {len(np.unique(labels))} clusters.")
            except Exception as e:
                print(f"An error occurred during clustering for '{perception}': {e}. Skipping.")

    # 7. Calculate Mean Scores per Cluster
    print("\n--- Calculating Mean Scores per Cluster ---")
    score_perceptions = set([col.split('_')[-2] for col in et_cells_clean_data.columns if col.startswith('score_')])

    for per in score_perceptions:
        cluster_id_col = f'{per}_ID'
        score_col = f'score_{per}'
        mean_score_col = f'mean_{per}'

        if cluster_id_col in et_cells_clean_data.columns and score_col in et_cells_clean_data.columns:
            et_cells_clean_data[mean_score_col] = et_cells_clean_data.groupby(cluster_id_col)[score_col].transform('mean')
            print(f"Calculated mean score for '{per}' clusters.")
        else:
            print(f"Skipping mean score calculation for '{per}': '{cluster_id_col}' or '{score_col}' column not found.")

    # 8. Final Data Cleaning and Export
    print("\n--- Final Data Cleaning and Export ---")
    # Drop PCA component columns as they are no longer needed
    pca_cols_to_drop = [col for col in et_cells_clean_data.columns if col.startswith('pc_')]
    et_cells_clean_data.drop(columns=pca_cols_to_drop, axis=1, inplace=True)

    # Save the result to a GeoPackage
    et_cells_clean_data.to_file(OUTPUT_GPKG_PATH, driver="GPKG")
    print(f"Processed data saved to: {OUTPUT_GPKG_PATH}")
    print("\nScript Finished Successfully.")