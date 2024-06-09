# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch

# First-party
from neural_lam import utils


class WeatherDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 6
    N_t = 6//subsample_step (= 6 for 1 day steps)
    dim_x = 774
    dim_y = 763
    N_grid = 147134
    d_features = 75
    d_atm = 6
    d_forcing = 8
    """

    def __init__(
        self,
        dataset_name,
        pred_length=4,
        split="train",
        subsample_step=1,
        standardize=True,
        subset=False,
        data_subset=None,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join(
            "data", dataset_name, "samples", split
        )

        member_file_regexp = (
            "rea_data_*.npy"
            if data_subset == "reanalysis"
            else (
                "ana_data_*.npy"
                if data_subset == "analysis"
                else "*_data_*.npy"
            )
        )
        sample_paths = glob.glob(
            os.path.join(self.sample_dir_path, member_file_regexp)
        )
        if split == "test":
            sample_paths = sorted(sample_paths)
        self.sample_names = [path.split("/")[-1][:-4] for path in sample_paths]
        # Now on form "yyymmddhh_mbrXXX"

        if subset:
            self.sample_names = self.sample_names[:20]  # Limit to 10 samples

        self.sample_length = pred_length + 2  # 2 init states
        self.subsample_step = subsample_step
        self.original_sample_length = (
            6 // self.subsample_step
        )  # 6 for 1 day steps
        assert (
            self.sample_length <= self.original_sample_length
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            (
                self.data_mean,
                self.data_std,
                self.forcing_mean,
                self.forcing_std,
            ) = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
                ds_stats["forcing_mean"],
                ds_stats["forcing_std"],
            )
            self.interior_mask = utils.load_mask(dataset_name, "cpu")[
                "interior_mask"
            ]

        # If subsample index should be sampled
        self.random_subsample = False

    def update_pred_length(self, new_length):
        """
        Update prediction length
        """
        assert (
            new_length + 2 <= self.original_sample_length
        ), "Requested prediction length too long"
        self.sample_length = new_length + 2

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        # === Sample ===
        sample_name = self.sample_names[idx]
        sample_path = os.path.join(self.sample_dir_path, f"{sample_name}.npy")
        try:
            full_sample = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            )  # (N_t', N_grid, d_features)
        except ValueError:
            print(f"Failed to load {sample_path}")

        # Only use every ss_step:th time step, sample which of ss_step
        # possible such time series
        if self.random_subsample:
            subsample_index = torch.randint(0, self.subsample_step, ()).item()
        else:
            subsample_index = 0
        subsample_end_index = self.original_sample_length * self.subsample_step
        sample = full_sample[
            subsample_index : subsample_end_index : self.subsample_step
        ]
        # (N_t, N_grid, d_features)

        # Uniformly sample time id to start sample from
        init_id = torch.randint(
            0, 1 + self.original_sample_length - self.sample_length, ()
        )
        sample = sample[init_id : (init_id + self.sample_length)]
        # (sample_length, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std
            sample = self.interior_mask * sample

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        # === Forcing features ===
        sample_datetime = sample_name[9:]

        forcing_path = os.path.join(
            self.sample_dir_path,
            f"forcing_{sample_datetime}.npy",
        )
        atm_forcing = torch.tensor(
            np.load(forcing_path), dtype=torch.float32
        )  # (N_t', N_grid, d_atm)

        if self.standardize:
            atm_forcing = (atm_forcing - self.forcing_mean) / self.forcing_std

        # Flatten and subsample atmospheric forcing
        atm_forcing = atm_forcing[
            subsample_index :: self.subsample_step
        ]  # (N_t, N_grid, d_atm)
        atm_forcing = atm_forcing[
            init_id : (init_id + self.sample_length)
        ]  # (sample_len, N_grid, 1)

        # Time of day and year
        dt_obj = dt.datetime.strptime(sample_datetime, "%Y%m%d")
        dt_obj = dt_obj + dt.timedelta(
            days=subsample_index
        )  # Offset for first index
        # Extract for initial step
        start_of_year = dt.datetime(dt_obj.year, 1, 1)
        init_seconds_into_year = (dt_obj - start_of_year).total_seconds()
        is_leap_year = dt_obj.year % 4 == 0 and (
            dt_obj.year % 100 != 0 or dt_obj.year % 400 == 0
        )
        if is_leap_year:
            seconds_in_year = 366 * 24 * 60 * 60
        else:
            seconds_in_year = 365 * 24 * 60 * 60

        # Add increments for all steps
        day_inc = (
            torch.arange(self.sample_length) * self.subsample_step
        )  # (sample_len,)
        second_into_year = (
            init_seconds_into_year + day_inc * 3600 * 24
        )  # (sample_len,)
        # can roll over to next year, ok because periodicity

        # Encode as sin/cos
        year_angle = (
            (second_into_year / seconds_in_year) * 2 * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 2)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, atm_forcing.shape[1], -1
        )  # (sample_len, N_grid, 2)

        # Put forcing features together
        forcing_features = torch.cat(
            (atm_forcing, datetime_forcing), dim=-1
        )  # (sample_len, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat(
            (
                forcing_features[:-2],
                forcing_features[1:-1],
                forcing_features[2:],
            ),
            dim=2,
        )  # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # Concat batch-static forcing here, if any
        forcing = forcing_windowed
        # (sample_len-2, N_grid, forcing_dim)

        return init_states, target_states, forcing
