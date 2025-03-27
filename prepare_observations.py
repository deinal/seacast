# Standard library
import argparse
import glob
import os

# Third-party
import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def prepare_sst(input_dir, output_dir):
    """
    Prepare SST data.
    """
    sst_dir = os.path.join(input_dir, "sst")
    ds = cm.open_dataset(
        dataset_id="SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_a",
        variables=["sea_surface_temperature"],
        minimum_longitude=-6,
        start_datetime="2024-07-01T00:00:00",
        end_datetime="2025-01-15T23:59:59",
    )
    ds["sea_surface_temperature"] = ds.adjusted_sea_surface_temperature.where(
        ~((ds.longitude < 2) & (ds.latitude > 42))
    )
    os.makedirs(sst_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sst.nc")
    ds.to_netcdf(output_file)


def prepare_sla(input_dir, output_dir):
    """
    Prepare SLA data.
      sla = sla_filtered + dac + ocean_tide + internal_tide

    For each day, combine the observations from all missions
    by stacking the time dimension into a new obs dimension,
    then assign a unique daily time coordinate.
    Finally, concat daily ds along the time dimension.
    """

    missions = {
        "AltiKa": 1,
        "Cryosat-2": 1,
        "HY-2B": 5,
        "Jason-3": 5,
        "Sentinel-3A": 5,
        "Sentinel-3B": 5,
        "Sentinel-6A": 5,
        "Swon": 5,
    }

    dates = pd.date_range("2024-07-01", "2025-01-15", freq="D")
    daily_ds_list = []

    # Loop over each day
    for date in tqdm(dates, desc="Processing days"):
        date_str = date.strftime("%Y%m%d")
        day_ds_parts = []
        for mission, freq in missions.items():
            month_str = date.strftime("%m")
            mission_dir = os.path.join(
                input_dir, "sla", mission, str(date.year), month_str
            )
            pattern = os.path.join(mission_dir, f"*{freq}hz_{date_str}_*.nc")
            files = glob.glob(pattern)
            for f in files:
                ds = xr.open_dataset(f)
                # Compute sla
                ds["sla"] = (
                    ds["sla_filtered"]
                    + ds["dac"]
                    + ds["ocean_tide"]
                    + ds["internal_tide"]
                )
                # Floor the time coordinate to the day
                ds = ds.assign_coords(time=ds["time"].dt.floor("D"))
                # Select only measurements for the current day
                ds_day = ds.where(ds.time == np.datetime64(date), drop=True)
                # Stack the time dimension into a new obs dimension
                ds_day = ds_day.stack(obs=("time",)).reset_index(
                    "obs", drop=True
                )
                # Keep only sla
                ds_day = ds_day[["sla"]]
                day_ds_parts.append(ds_day)
                ds.close()

        # Concatenate along the new obs dim
        day_ds = xr.concat(day_ds_parts, dim="obs", combine_attrs="drop")
        # Assign a unique time coordinate for the day
        day_ds = day_ds.expand_dims(time=[np.datetime64(date, "ns")])
        daily_ds_list.append(day_ds)

    # Pad to max number of obs per day
    max_obs = max(ds.dims["obs"] for ds in daily_ds_list)
    padded_ds_list = []
    for ds in daily_ds_list:
        pad_width = max_obs - ds.dims["obs"]
        if pad_width > 0:
            ds_padded = ds.pad(obs=(0, pad_width), constant_values=np.nan)
        else:
            ds_padded = ds
        padded_ds_list.append(ds_padded)

    # Concatenate daily datasets along the time dimension
    ds_all = xr.concat(padded_ds_list, dim="time")
    ds_all = ds_all.sortby("time")
    output_file = os.path.join(output_dir, "sla.nc")
    ds_all.to_netcdf(output_file)
    print(ds_all)
    print(f"Merged SLA dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare observation data.")
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        choices=["sst", "sla"],
        required=True,
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/mediterranean/raw"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/mediterranean/observations"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if "sst" in args.data:
        prepare_sst(args.input_dir, args.output_dir)
    if "sla" in args.data:
        prepare_sla(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
