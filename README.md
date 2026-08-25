
# SeaCast

[![arXiv](https://img.shields.io/badge/arXiv-2506.23900-b31b1b.svg)](https://arxiv.org/abs/2506.23900) [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/deinal/seacast-data)

<p align="middle">
    <img src="figures/hi_graph.png" width="700">
</p>

SeaCast is based on [Neural-LAM](https://github.com/mllam/neural-lam) (data-driven limited area weather forecasting). This repository contains mesh variations similar to:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2024)](https://arxiv.org/abs/2406.04759).

## Citation

```
@article{holmberg2025accurate,
  title={Accurate Mediterranean Sea forecasting via graph-based deep learning},
  author={Holmberg, Daniel and Clementi, Emanuela and Epicoco, Italo and Roos, Teemu},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={45051},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

```
@inproceedings{holmberg2024regional,
    title={Regional Ocean Forecasting with Hierarchical Graph Neural Networks},
    author={Holmberg, Daniel and Clementi, Emanuela and Roos, Teemu},
    booktitle={NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning},
    year={2024}
}
```

## Dependencies

SeaCast was trained using Python 3.10 and
- `torch==2.4.1`
- `pytorch-lightning==2.4.0`
- `torch_geometric==2.5.3`

Complete set of packages can be installed with `pip install -r requirements.txt`.

## Data

### Quickstart with preprocessed data

Preprocessed data used in the associated article is stored on Hugging Face (https://huggingface.co/datasets/deinal/seacast-data). This data can be downloaded using their CLI:

```
pip install huggingface_hub
huggingface-cli download deinal/seacast-data --repo-type dataset --local-dir .  --exclude "README.md" --exclude ".gitattributes"
```

The full data folder (4.77 TB) including all training and validation files is stored on LUMI object storage (https://462000711.lumidata.eu/seacast-data). You can fetch all of them with [files.txt](https://drive.google.com/file/d/1rG9vUiTg0jmikFJz8pxVnHoault1fNBL/view?usp=sharing) by running `wget -i files.txt`.

The following subsections cover how the data was originally fetched and preprocessed.

### Download instructions

1. Create accounts on Copernicus marine (https://marine.copernicus.eu) and climate data store (https://cds.climate.copernicus.eu).

2. Log in to the marine service on your machine using the Python client `copernicusmarine login`, and [set up climate credentials](https://cds.climate.copernicus.eu/api-how-to) to access atmospheric data.

3. Then download all the training data:
```
python download_data.py -d reanalysis -s 1987-01-01 -e 2022-07-31 --static
python download_data.py -d analysis -s 2021-11-01 -e 2025-01-14
python download_data.py -d era5 -s 1987-01-01 -e 2024-06-30
```

4. Daily forecasts were fetched  with the ECMWF [open data client](https://pypi.org/project/ecmwf-opendata/) and CMEMS python client using a daily cronjob:
```
0 21 * * * python download_data.py --forecast >> forecasts.log 2>&1
```

5. Observations (assumes SLA and in situ data stored in raw dir)
```
python prepare_observations.py -d sst mhw sla in_situ
```

### State preparation

Mediterranean reanalysis
```
python prepare_states.py -d data/mediterranean/raw/reanalysis -o data/mediterranean/samples/train -n 6 -p rea_data -s 1987-01-01 -e 2021-12-31
```

Mediterranean analysis
```
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/train -n 6 -p ana_data -s 2022-01-01 -e 2023-12-31
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/val -n 6 -p ana_data -s 2024-01-01 -e 2024-06-30
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/test -n 17 -p ana_data -s 2024-07-01 -e 2024-01-14 --forecast
```

ERA5
```
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/train -n 6 -p forcing -s 1987-01-01 -e 2023-12-31
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/val -n 6 -p forcing -s 2024-01-01 -e 2024-06-30
```

Forecast data
```
python prepare_states.py -d data/mediterranean/raw/forecast -o data/mediterranean/samples/test -p for_data -s 2024-07-03 -e 2024-12-31 --forecast
python prepare_states.py -d data/mediterranean/raw/ens -o data/mediterranean/samples/test -p ens_forcing -s 2024-07-03 -e 2024-12-31 --forecast
python prepare_states.py -d data/mediterranean/raw/aifs -o data/mediterranean/samples/test -p aifs_forcing -s 2024-07-03 -e 2024-12-31 --forecast
```

Compute wind stress components from wind velocity components
```
python compute_wind_stress.py
```

### Create static features

```
python create_grid_features.py --dataset mediterranean
```
Stored in the `static` directory of your dataset.

### Calculate dataset statistics

```
python create_parameter_weights.py --dataset mediterranean
```
Stored in the `static` directory of your dataset.

## Training

### Create model graph

```
python create_mesh.py --dataset mediterranean --graph hierarchical --levels 3 --hierarchical 1
```
Stored in a new directory `graphs/hierarchical`.

### Logging

The project is compatible with weights and biases (https://wandb.ai).
```
wandb login
```
To log things locally, run:
```
wandb off
```

### Train models

SeaCast is first trained on reanalysis data, using 8 nodes:
```
python train_model.py \
  --n_nodes 8 \
  --n_workers 4 \
  --epochs 200 \
  --lr 0.001 \
  --batch_size 1 \
  --step_length 1 \
  --ar_steps 1 \
  --optimizer adamw \
  --scheduler cosine \
  --processor_layers 3 \
  --hidden_dim 256 \
  --model hi_lam \
  --graph hierarchical \
  --precision bf16-mixed \
  --data_subset reanalysis \
  --run_id all_base
```
To train the 10y model from the paper add the flag `--start_date 20140101`.

Models are finetuned on analysis data, using 1 node:
```
python train_model.py \
  --load saved_models/hi_lam-3x256-all_base/min_val_loss.ckpt \
  --n_nodes 1 \
  --n_workers 4 \
  --epochs 30 \
  --lr 1e-5 \
  --initial_lr 1e-7 \
  --batch_size 1 \
  --step_length 1 \
  --ar_steps 3 \
  --finetune_start 0.34 \
  --optimizer adamw \
  --scheduler cosine \
  --processor_layers 3 \
  --hidden_dim 256 \
  --model hi_lam \
  --graph hierarchical \
  --precision bf16-mixed \
  --data_subset analysis \
  --run_id all_final
```

For a full list of possible training options, check `python train_model.py --help`.

## Evaluation

To produce the predictions on 1 GPU use `--eval test` and specify output directory (under `data`) with `--run_id seacast`:
```
python train_model.py \
  --data_subset forecast \
  --forcing_prefix aifs_forcing \
  --n_workers 4 \
  --batch_size 1 \
  --step_length 1 \
  --model hi_lam \
  --graph hierarchical \
  --processor_layers 3 \
  --hidden_dim 256 \
  --n_example_pred 1 \
  --store_pred 1 \
  --eval test \
  --precision bf16-mixed \
  --load model_weights/final.ckpt \
  --run_id seacast
```

To instead try ENS forcing, use `--forcing_prefix ens_forcing`, to use analysis initial conditions use `--data_subset analysis`, or to permute the atmospheric forcing use `--permute_forcing tau_u tau_v t2m msl` (or a subset of forcing variables).

For evaluation on analysis fields run:
```
python -u calculate_metrics.py --n_workers 10 --forecast seacast
```

For evaluation on observations run:
```
python compare_sst.py --forecast seacast
python compare_mhw.py --forecast seacast
python compare_sla.py --forecast seacast
python compare_in_situ.py --forecast seacast
```
Note that for SLA and in-stu dask will parallelize to all available cores by default.

Results can be plotted with
```
python plot_results.py
```
and for a more exhaustive set of plots run:
```
python plot_metrics.py \
  --plot_group_bias \
  --plot_forecast \
  --plot_rmse \
  --plot_acc \
  --plot_forecast_vertical \
  --plot_scorecard \
  --plot_rmse_depth \
  --plot_norm_rmse_diff \
  --plot_spatial_rmse_diff \
  --plot_vertical_rmse_diff \
  --file data/mediterranean/predictions/seacast/for_data_20241001.npy
```

## File structure

### Code

Scripts to execute data retrieval, preprocessing, training, etc. are all located at the root of the repository, and the source code is in the `neural_lam` directory.

### Data

It is possible to store multiple datasets in the `data` directory. Each dataset contains a set of files with static features and a set of samples. Example below:

```
data
├── mediterranean
│   ├── samples                             - Directory with data samples
│   │   ├── train                           - Training data
│   │   │   ├── ana_data_20211103.npy       - Analysis sample
│   │   │   ├── ana_data_20211104.npy
│   │   │   ├── ...
│   │   │   ├── forcing_19870103.npy        - Atmospheric forcing
│   │   │   ├── forcing_19870104.npy
│   │   │   ├── ...
│   │   │   ├── rea_data_20211103.npy       - Reanalysis sample
│   │   │   ├── rea_data_20211104.npy
│   │   │   ├── ...
│   │   │   └── rea_data_20211031.npy
│   │   ├── val                             - Validation data
│   │   └── test                            - Test data
│   └── static                              - Directory with graph information and static features
│       ├── bathy_mask.nc                   - Full bathymetry mask (part of dataset)
│       ├── nwp_xy.npy                      - Coordinates of grid nodes (part of dataset)
│       ├── coordinates.npy                 - Lat-lon coordinates of grid nodes (part of dataset)
│       ├── sea_depth.npy                   - Sea floor depth below geoid (part of dataset)
│       ├── sea_mask.npy                    - Sea binary mask (part of dataset)
│       ├── sea_topography.npy              - Mean dynamic topography (part of dataset)
│       ├── boundary_mask.npy               - Boundary mask (part of dataset)
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)
│       ├── forcing_mean.pt                 - Means of atmospheric forcing (create_parameter_weights.py)
│       ├── forcing_std.pt                  - Std.-dev. of atmospheric forcing (create_parameter_weights.py)
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)
├── baltic
├── ...
└── datasetN
```

## Development

GitHub actions are implemented for code checks. Run before commits:
```
pre-commit run --all-files
```
from the root directory of the repository.
