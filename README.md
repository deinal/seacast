
# SeaCast

<p align="middle">
    <img src="figures/hi_graph.png" width="700">
</p>

The code is based on [Neural-LAM](https://github.com/joeloskarsson/neural-lam): a repository of graph-based neural weather prediction models for limited area modeling. Specifically, the [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) branch.

The repository contains three meshgraphnet variations:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

## Dependencies

SeaCast was trained using Python 3.10 and
- `torch==2.2.2`
- `pytorch-lightning==2.2.0`
- `torch_geometric==2.5.3`

Complete set of packages can be installed with `pip install -r requirements.txt`.

## Data

### Download instructions

1. Create accounts on Copernicus marine (https://marine.copernicus.eu) and climate data store (https://cds.climate.copernicus.eu).

2. Log in to the marine service on your machine using the Python client `copernicusmarine login`, and [set up climate credentials](https://cds.climate.copernicus.eu/api-how-to) to access atmospheric data.

3. Then download all the training data:
```
python download_data.py -d reanalysis -s 1987-01-01 -e 2022-07-31
python download_data.py -d analysis -s 2021-11-01 -e 2024-07-31
python download_data.py -d era5 -s 1987-01-01 -e 2024-05-31
```

4. Daily forecasts were fetched  with the ECMWF [open data client](https://pypi.org/project/ecmwf-opendata/) for the months of June and July 2024 using a cronjob:
```
0 21 * * * python download_data.py --forecast >> forecasts.log 2>&1
```

### State preparation

Mediterranean reanalysis
```
python prepare_states.py -d data/mediterranean/raw/reanalysis -o data/mediterranean/samples/train -n 6 -p rea_data -s 1987-01-01 -e 2021-11-30
python prepare_states.py -d data/mediterranean/raw/reanalysis -o data/mediterranean/samples/val -n 6 -p rea_data -s 2021-11-01 -e 2021-12-31
python prepare_states.py -d data/mediterranean/raw/reanalysis -o data/mediterranean/samples/test -n 16 -p rea_data -s 2022-01-01 -e 2022-07-31
```

Mediterranean analysis
```
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/train -n 6 -p ana_data -s 2021-11-01 -e 2024-03-31
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/val -n 6 -p ana_data -s 2024-04-01 -e 2024-05-31
python prepare_states.py -d data/mediterranean/raw/analysis -o data/mediterranean/samples/test -n 11 -p ana_data -s 2024-06-01 -e 2024-07-31
```

ERA5
```
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/train -n 6 -p forcing -s 1987-01-01 -e 2024-03-31
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/val -n 6 -p forcing -s 2021-11-01 -e 2021-12-31
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/val -n 6 -p forcing -s 2024-04-01 -e 2024-05-31
python prepare_states.py -d data/mediterranean/raw/era5 -o data/mediterranean/samples/test -n 16 -p forcing -s 2022-01-01 -e 2022-07-31
```

ECMWF HRES
```
python prepare_states.py -d data/mediterranean/raw/hres -o data/mediterranean/samples/test -p forcing -s 2024-06-01 -e 2024-07-31 --forecast_forcing
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

SeaCast was trained on 4 nodes with 8 GPUs each:
```
python train_model.py \
  --data_subset reanalysis \
  --epochs 200 \
  --n_workers 4 \
  --batch_size 1 \
  --step_length 1 \
  --ar_steps 4 \
  --lr 0.001 \
  --scheduler cosine \
  --finetune_start 0.6 \
  --model hi_lam \
  --graph hierarchical \
  --processor_layers 4 \
  --hidden_dim 128 \
  --n_nodes 4
```
For finetuing update arguments `--data_subset analysis`, `--epochs 50` and `--lr 0.0001`.

For a full list of possible training options, check `python train_model.py --help`.

## Evaluation

SeaCast was evaluated on 1 GPU using `--eval test`, and by choosing the correct data subset + loading the appropriate model:
```
python train_model.py \
  --data_subset reanalysis \
  --n_workers 4 \
  --batch_size 1 \
  --step_length 1 \
  --model hi_lam \
  --graph hierarchical \
  --processor_layers 4 \
  --hidden_dim 128 \
  --n_example_pred 1 \
  --eval test \
  --load saved_models/hi_lam-4x128-06_26_19-6986/last.ckpt
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
│       ├── strait_mask                     - Boundary mask (part of dataset)
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)
│       ├── forcing_stats.pt                - Mean and std.-dev. of forcing (create_parameter_weights.py)
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)
├── baltic
├── ...
└── datasetN
```

### Graphs

The `graphs` directory contains generated graph structures that can be used by different graph-based models. Refer to https://github.com/mllam/neural-lam for more details.

## Development

GitHub actions are implemented for code checks. Run before commits:
```
pre-commit run --all-files
```
from the root directory of the repository.

## Cite

Neural-LAM
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
