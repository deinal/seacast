# Standard library
import argparse
import os
from glob import glob

# Third-party
import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm

# First-party
from neural_lam import constants


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
    ds["sea_surface_temperature"] = ds.sea_surface_temperature.where(
        ~((ds.longitude < 2) & (ds.latitude > 42))
    )
    os.makedirs(sst_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sst.nc")
    ds.to_netcdf(output_file)


def pad_dataset_list(ds_list, dim, constant_values=np.nan):
    """
    Pad each ds along the given dimension so that all have the same size.

    Parameters:
      ds_list: list of ds objects.
      dim: the dimension along which to pad (e.g., obs or file).
      constant_values: the value to pad with (default: nan).

    Returns:
      A list of ds objects padded along the specified dim to the same size.
    """
    target_size = max(ds.sizes[dim] for ds in ds_list)
    padded_list = []
    for ds in ds_list:
        current_size = ds.sizes[dim]
        pad_width = target_size - current_size
        if pad_width > 0:
            ds = ds.pad(
                **{dim: (0, pad_width)}, constant_values=constant_values
            )
        padded_list.append(ds)
    return padded_list


def prepare_sla(input_dir, output_dir):
    """
    Prepare SLA data.
      sla = sla_filtered + dac + ocean_tide + internal_tide

    For each day, combine the observations from all missions
    by stacking the time dimension into a new obs dimension,
    then assign a unique daily time coordinate. Finally, pad
    and concatenate the daily datasets along time.
    """

    missions = [
        "HY-2B",
        "Jason-3",
        "Sentinel-3A",
        "Sentinel-3B",
        "Sentinel-6A",
        "Swon",
    ]

    dates = pd.date_range("2024-07-01", "2025-01-15", freq="D")
    daily_ds_list = []

    # Process each day
    for date in tqdm(dates, desc="Processing days"):
        date_str = date.strftime("%Y%m%d")
        day_ds_parts = []
        for mission in missions:
            month_str = date.strftime("%m")
            mission_dir = os.path.join(
                input_dir, "sla", mission, str(date.year), month_str
            )
            pattern = os.path.join(mission_dir, f"*5hz_{date_str}_*.nc")
            files = glob(pattern)
            for f in files:
                ds = xr.open_dataset(f)
                # Compute SLA
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
                # Keep only the SLA variable
                ds_day = ds_day[["sla"]]
                day_ds_parts.append(ds_day)
                ds.close()

        # Pad each file's obs dimension within the day to have the same size
        padded_day_ds = pad_dataset_list(
            day_ds_parts, "obs", constant_values=np.nan
        )
        # Concatenate along a new file dimension
        day_ds = xr.concat(padded_day_ds, dim="file", combine_attrs="drop")
        # Assign a unique time coordinate for the day
        day_ds = day_ds.expand_dims(time=[np.datetime64(date, "ns")])
        daily_ds_list.append(day_ds)

    # Pad daily datasets along the obs dimension across days
    padded_ds_obs = pad_dataset_list(
        daily_ds_list, "obs", constant_values=np.nan
    )
    # Pad daily datasets along the file dimension across days
    padded_ds_files = pad_dataset_list(
        padded_ds_obs, "file", constant_values=np.nan
    )

    # Concatenate daily datasets along the time dimension
    ds_all = xr.concat(padded_ds_files, dim="time")
    ds_all = ds_all.sortby("time")
    output_file = os.path.join(output_dir, "sla.nc")
    ds_all.to_netcdf(output_file)
    print(f"Merged SLA dataset saved to {output_file}")


def compute_grid_bins(surface_mask):
    """
    Compute bin edges for latitude and longitude from the surface mask.
    """
    # Get unique grid cell centers
    lat_centers = np.array(surface_mask.latitude.values)
    lon_centers = np.array(surface_mask.longitude.values)

    # Sort if not already sorted
    lat_centers = np.sort(lat_centers)
    lon_centers = np.sort(lon_centers)

    # Calculate bin edges as midpoints between adjacent centers
    lat_edges = np.concatenate(
        (
            [lat_centers[0] - (lat_centers[1] - lat_centers[0]) / 2],
            (lat_centers[:-1] + lat_centers[1:]) / 2,
            [lat_centers[-1] + (lat_centers[-1] - lat_centers[-2]) / 2],
        )
    )

    lon_edges = np.concatenate(
        (
            [lon_centers[0] - (lon_centers[1] - lon_centers[0]) / 2],
            (lon_centers[:-1] + lon_centers[1:]) / 2,
            [lon_centers[-1] + (lon_centers[-1] - lon_centers[-2]) / 2],
        )
    )

    return lat_edges, lon_edges


@delayed
def process_file(f, lat_edges, lon_edges, depth_edges, variables, date):
    """Delayed function to process a single file."""
    ds = xr.open_dataset(f)
    # Skip file if none of the variables are present
    if not any(v in ds.data_vars for v in variables):
        return None
    # Keep only quality checked data
    present_meas = set(variables) & set(ds.data_vars)
    for meas in present_meas:
        qc_var = f"{meas}_QC" if f"{meas}_QC" in ds.data_vars else "QCflag"
        ds[meas] = ds[meas].where(ds[qc_var] == 1)

    try:
        ds = ds[present_meas]
    except KeyError:
        return None

    source = ds.attrs.get("source", "unknown")
    if source == "unknown":
        return None

    # Convert dataset to df
    df_file = ds.to_dataframe().reset_index()
    if "DEPH" not in df_file.columns:
        df_file["DEPH"] = df_file["PRES"] / 1.019716

    # Create latitude, longitude, and depth bins
    df_file["lat_bin"] = pd.cut(
        df_file["LATITUDE"], bins=lat_edges, include_lowest=True
    )
    df_file["lon_bin"] = pd.cut(
        df_file["LONGITUDE"], bins=lon_edges, include_lowest=True
    )
    df_file["depth_bin"] = pd.cut(
        df_file["DEPH"], bins=depth_edges, include_lowest=True
    )

    # Group by the grid cell bins and depth bin
    group_cols = ["lat_bin", "lon_bin", "depth_bin"]
    grouped = df_file.groupby(group_cols, observed=False)

    dfs_local = []
    for (lat_bin, lon_bin, depth_bin), group in grouped:
        if group.empty:
            continue
        # Check if all measurement variables are nan in this group
        if group[list(present_meas)].isna().all().all():
            continue
        df_mean = group.mean(skipna=True, numeric_only=True)
        df_mean["date"] = date
        df_mean["source"] = source
        dfs_local.append(df_mean)

    if dfs_local:
        return pd.concat(dfs_local, axis=1).T
    else:
        return None


def prepare_in_situ(input_dir, output_dir):
    data_dir = os.path.join(input_dir, "in_situ")
    dates = pd.date_range("2024-07-01", "2025-01-15", freq="D")
    variables = ["EWCT", "NSCT", "PSAL", "TEMP"]
    delayed_results = []

    # Load bathymetry to get the surface mask
    bathy_path = os.path.join(
        "data", "mediterranean", "static", "bathy_mask.nc"
    )
    bathy_data = xr.load_dataset(bathy_path)
    sea_mask = bathy_data.where(bathy_data.mask, drop=True).mask
    surface_mask = sea_mask.isel(depth=0)

    # Compute latitude and longitude bin edges from the surface mask
    lat_edges, lon_edges = compute_grid_bins(surface_mask)

    # Define depth bins using the depth centers
    centers = constants.DEPTHS
    depth_edges = (
        [0]
        + [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
        + [200]
    )
    print("Depth edges:", depth_edges)

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        date_dir = os.path.join(data_dir, date_str)
        files = glob(os.path.join(date_dir, "*.nc"))
        for f in files:
            delayed_results.append(
                process_file(
                    f, lat_edges, lon_edges, depth_edges, variables, date
                )
            )

    # Compute all the delayed tasks in parallel
    with ProgressBar():
        results = compute(*delayed_results, scheduler="processes")

    # Filter results
    dfs = [r for r in results if r is not None]
    df_all = pd.concat(dfs, axis=0)

    # Define columns to keep
    cols_to_keep = [
        "date",
        "LATITUDE",
        "LONGITUDE",
        "DEPH",
        "source",
    ] + variables
    df_all = df_all[cols_to_keep]
    df_all = df_all.rename(columns={"DEPH": "depth"})
    df_all.columns = df_all.columns.str.lower()

    df_all = df_all.sort_values("date")
    df_all = df_all.reset_index(drop=True)

    ds_out = xr.Dataset.from_dataframe(df_all)
    output_file = os.path.join(output_dir, "in_situ.nc")
    ds_out.to_netcdf(output_file)
    print(f"Merged in situ data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare observation data.")
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        choices=["sst", "sla", "in_situ"],
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
    if "in_situ" in args.data:
        prepare_in_situ(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
