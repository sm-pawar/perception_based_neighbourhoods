# Rethinking neighbourhood boundaries for urban planning: A data-driven framework for perception-based delineation

This repository contains supporting files and scripts used in this research for delineating urban neighbourhoods based on perception data.

## Project Structure

```
Perception_based_neighbourhoods/
├── raw_data/
│   ├── ET_cells_glasgow.shp
│   ├── glasgow_bounds.shp
│   ├── glasgow_open_built.shp
│   └── glasgow_svi_grid.shp
├── svi_module/
│   ├── output_data/
│   │   ├── svi_info.csv               # Demo data for next step
│   │   └── images/                    # Downloaded street view images
│   ├── get_svi_data.py                # Download Places Pulse data from Figshare
│   └── trueskill_score.py             # Generate TrueSkill scores for each image
├── perception_module/
│   ├── output_data/
│   │   └── glasgow_perception.nc      # Demo data for next step
│   ├── trained_models/                # Pre-trained perception models
│   └── pred.py                        # Predict perceptions from images
└── cluster_module/
    ├── output_data/
    │   └── clusters.shp               # Final clustered neighbourhoods
    └── cluster_perceptions.py         # Generate neighbourhood clusters
```

## Prerequisites

- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  - geopandas
  - pandas
  - numpy
  - xarray
  - scikit-learn
  - [Add other dependencies as needed]

## Step-by-Step Guide

### Step 1: Download Street View Images (SVI Module)

The first module downloads street view images based on the Glasgow grid.

**Input:**
- `/raw_data/os_built_extent/glasgow_open_built_areas.shp` - Grid defining sampling locations for street view images

**Script:**
```bash
python svi_module/get_svi_data.py
```

**Output:**
- `svi_module/output_data/svi_info.csv` - Metadata for downloaded images (image IDs, coordinates, etc.)
- `svi_module/output_data/images/` - Downloaded street view images

**Note:** Demo data is already provided in `svi_module/output_data/` to enable testing subsequent steps without running the full download.

---

### Step 2: Predict Perceptions (Perception Module)

This module uses pre-trained models to predict perceptual qualities (e.g., safety, beauty, liveliness) for each street view image.

**Input:**
- `svi_module/output_data/svi_info.csv` - Image metadata from Step 1
- `perception_module/pre_trained_models/` - Pre-trained perception prediction models

**Script:**
```bash
python -m perception_module.pred \
    --model-weights ./trained_models/best_model_val_efficientnet_b4_beautiful.pt \
    --backbone efficientnet_b4 \
    --image-dir ../svi_module/output_data/images \
    --perception-type beautiful \
    --output-file efficientnet_b4_beautiful_glasgow_embeddings.nc \
    --extract-embeddings 
```

**Output:**
- `perception_module/output_data/efficientnet_b4_beautiful_glasgow_embeddings.nc` - NetCDF file containing perception scores and embeddings for all images. 

**Note:** Demo data is already provided in `perception_module/output_data/` to enable testing the clustering step.

**Note:**  The instruction to train model using Place Pulse databased are available [here](/perception_module/readme.md). 

---

### Step 3: Generate Neighbourhood Clusters (Cluster Module)

The final module clusters locations based on their perception profiles to delineate perceptual neighbourhoods.

**Input:**
- `perception_module/output_data/glasgow_perception.nc` - Perception scores from Step 2

**Script:**
```bash
python cluster_module/cluster_perceptions.py
```

**Output:**
- `cluster_module/output_data/clusters.shp` - Shapefile of delineated neighbourhood boundaries based on perception similarity

---

## Quick Start with Demo Data

If you want to test individual modules without running the full pipeline, demo data is provided in each module's `output_data/` folder:

1. **Test perception prediction only:**
   ```bash
   python perception_module/pred.py
   ```
   Uses demo `svi_info.csv` from `svi_module/output_data/`

2. **Test clustering only:**
   ```bash
   python cluster_module/cluster_perceptions.py
   ```
   Uses demo `glasgow_perception.nc` from `perception_module/output_data/`

## Citation

If you use this code or methodology in your research, please cite:

```
[Add your citation here]
```

## License

[Add license information]

## Contact

[Add contact information for questions]