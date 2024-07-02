# Standard library
import argparse
import calendar
import os
from datetime import datetime, timedelta

# Third-party
import cdsapi
import cfgrib
import copernicusmarine as cm
import ecmwf.opendata as eo
import numpy as np
import pandas as pd
import scipy
import xarray as xr

# First-party
from neural_lam import constants


def load_mask(path):
    """
    Load bathymetry mask.

    Args:
    path (str): Path to the bathymetry mask file.

    Returns:
    mask (xarray.Dataset): Bathymetry mask.
    """
    if os.path.exists(path):
        print("Bathymetry mask file found. Loading from file.")
        mask = xr.load_dataset(path).mask
    else:
        print("Bathymetry mask file not found. Downloading...")
        bathy_data = cm.open_dataset(
            dataset_id="cmems_mod_med_phy_my_4.2km_static",
            dataset_part="bathy",
            dataset_version="202211",
            service="static-arco",
            variables=["mask"],
            minimum_depth=constants.DEPTHS[0],
            maximum_depth=constants.DEPTHS[-1],
        )
        bathy_data["longitude"] = bathy_data.longitude.where(
            (bathy_data.longitude < -1e-9) | (bathy_data.longitude > 1e-9),
            other=0.0,
        )
        mask = bathy_data.mask.sel(depth=constants.DEPTHS)
        mask.to_netcdf(path)

    return mask


def select(dataset, mask):
    """
    Select masked volume.

    Args:
    dataset (xarray.Dataset): Input dataset.
    mask (xarray.Dataset): Bathymetry mask.

    Returns:
    mask (xarray.Dataset): Masked dataset.
    """
    # Fix longitude mismatches close to zero
    dataset["longitude"] = dataset.longitude.where(
        (dataset.longitude < -1e-9) | (dataset.longitude > 1e-9), other=0.0
    )

    # Select depth levels, and masked area
    if hasattr(dataset, "depth"):
        dataset = dataset.sel(depth=constants.DEPTHS)
        dataset = dataset.where(mask, drop=True)
    else:
        dataset = dataset.where(mask.isel(depth=0), drop=True)

    # Uncomment to coarsen grid
    # dataset = dataset.coarsen(longitude=2, boundary="pad").mean()
    # dataset = dataset.coarsen(latitude=2, boundary="pad").mean()
    return dataset


def download_static(path_prefix, mask):
    """
    Download static data as numpy files.

    Args:
    path_prefix (str): Path to store.
    mask (xarray.Dataset): Bathymetry mask.
    """
    # Download ocean depth and mask
    bathy_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="bathy",
        dataset_version="202211",
        service="static-arco",
        variables=["deptho", "mask"],
        minimum_depth=constants.DEPTHS[0],
        maximum_depth=constants.DEPTHS[-1],
    )
    bathy_data = select(bathy_data, mask)

    sea_depth = bathy_data.deptho.isel(depth=0)
    np.save(f"{path_prefix}/sea_depth.npy", np.nan_to_num(sea_depth, nan=0.0))

    # Store bathymetry mask
    sea_mask = ~np.isnan(bathy_data.mask)
    np.save(f"{path_prefix}/sea_mask.npy", sea_mask)

    # Forcing mask for the Strait of Gibraltar
    border_mask = np.where(
        bathy_data.mask.longitude < -5.2, bathy_data.mask, np.nan
    )
    border_mask = ~np.isnan(border_mask)
    np.save(f"{path_prefix}/strait_mask.npy", border_mask)

    y_indices, x_indices = np.indices(sea_mask.shape[1:])
    nwp_xy = np.stack([x_indices, y_indices])
    np.save(f"{path_prefix}/nwp_xy.npy", nwp_xy)

    lat = bathy_data.latitude
    lon = bathy_data.longitude
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    coordinates = np.stack([lat_mesh, lon_mesh])  # (2, h, w)
    np.save(f"{path_prefix}/coordinates.npy", coordinates)

    # Download mean dynamic topography
    mdt_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="mdt",
        dataset_version="202211",
        service="static-arco",
        variables=["mdt"],
    )
    mdt_data = select(mdt_data, mask)
    np.save(
        f"{path_prefix}/sea_topography.npy",
        np.nan_to_num(mdt_data.mdt, nan=0.0),
    )

    # Download coordinate data
    coord_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="coords",
        dataset_version="202211",
        service="static-arco",
        variables=["e1t"],
    )
    coord_data = select(coord_data, mask)
    grid_weights = coord_data.e1t / coord_data.e1t.mean()
    np.save(
        f"{path_prefix}/grid_weights.npy", np.nan_to_num(grid_weights, nan=0.0)
    )


def download_data(
    start_date,
    end_date,
    datasets,
    version,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save daily physics data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    datasets (dict): Datasets to download.
    version (str): Dataset version.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask
    """

    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    current_date = start_date
    while current_date <= end_date:
        # Calculate the first and last day of the current month
        first_day = current_date.replace(day=1)
        last_day = current_date.replace(
            day=calendar.monthrange(current_date.year, current_date.month)[1]
        )

        # Format the start and end datetime strings for the month
        start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
        end_datetime = last_day.strftime("%Y-%m-%dT00:00:00")

        # Request data for the whole month
        month_data = {}
        for dataset_id, variables in datasets.items():
            ds = cm.open_dataset(
                dataset_id=dataset_id,
                dataset_version=version,
                dataset_part="default",
                service="arco-geo-series",
                variables=variables,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                minimum_depth=constants.DEPTHS[0],
                maximum_depth=constants.DEPTHS[-1],
            )
            month_data[dataset_id] = select(ds, mask)
            ds.close()

        # Process and save daily data
        for day in range((last_day - first_day).days + 1):
            day_date = first_day + timedelta(days=day)
            filename = f"{path_prefix}/{day_date.strftime('%Y%m%d')}.npy"
            all_data = []
            for dataset_id in datasets:
                dataset = month_data[dataset_id]
                for var in datasets[dataset_id]:
                    daily_data = dataset[var].isel(time=day).values
                    if var == "bottomT":
                        daily_data = daily_data[:, :, 0]
                    if len(daily_data.shape) == 2:
                        daily_data = daily_data[np.newaxis, ...]
                    daily_data = daily_data.transpose(1, 2, 0)  # h, w, f
                    all_data.append(daily_data)

            # Concatenate all data along a new axis
            combined_data = np.concatenate(all_data, axis=-1)
            clean_data = np.nan_to_num(combined_data, nan=0.0)

            # Select only sea grid points
            sea_data = clean_data[grid_mask, :]  # n_grid, f

            # Save combined data as one numpy file
            np.save(filename, sea_data)
            print(f"Saved data to {filename}")

        # Increment the next month
        current_date = last_day + timedelta(days=1)


def download_forecast(
    start_date,
    end_date,
    datasets,
    version,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save forecast data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    datasets (dict): Datasets to download.
    version (str): Dataset version.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.npy"

    all_data = []
    for dataset_id, variables in datasets.items():
        # Load ocean physics dataset for all dates at once
        dataset = cm.open_dataset(
            dataset_id=dataset_id,
            dataset_version=version,
            dataset_part="default",
            service="arco-geo-series",
            variables=variables,
            start_datetime=start_date.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=end_date.strftime("%Y-%m-%dT00:00:00"),
            minimum_depth=constants.DEPTHS[0],
            maximum_depth=constants.DEPTHS[-1],
        )

        dataset = select(dataset, mask)
        for var in variables:
            data = dataset[var].values
            if var == "bottomT":
                data = data[:, :, :, 0]
            if len(data.shape) == 3:
                data = data[:, np.newaxis, ...]  # t, 1, h, w
            data = data.transpose(0, 2, 3, 1)  # t, h, w, f
            all_data.append(data)

        dataset.close()

    # Concatenate all data along the feature dimension
    combined_data = np.concatenate(all_data, axis=-1)
    combined_data = np.nan_to_num(combined_data, nan=0.0)

    # Select only sea grid points
    sea_data = combined_data[:, grid_mask, :]  # t, n_grid, f

    # Save combined data as one numpy file
    np.save(filename, sea_data)
    print(f"Saved forecast data to {filename}")


def download_era5(
    start_date,
    end_date,
    request_variables,
    ds_variables,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save daily ERA5 data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    request_variables (list): List of variables to request from cds.
    ds_variables (list): List of variables in the dataset.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask.
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    client = cdsapi.Client()
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month

        filename = f"{path_prefix}/{current_date.strftime('%Y%m')}.nc"
        if os.path.isfile(filename):
            if month == 7:
                current_date = current_date.replace(
                    year=year + 1, month=1, day=1
                )
            else:
                current_date = current_date.replace(month=month + 6, day=1)
            continue

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "format": "netcdf",
                "product_type": "reanalysis",
                "variable": request_variables,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(month, month + 6)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
                "area": [
                    mask.latitude.max().item(),
                    mask.longitude.min().item(),
                    mask.latitude.min().item(),
                    mask.longitude.max().item(),
                ],
            },
            filename,
        )
        print(f"Downloaded {filename}")

        # Load the data and average to daily
        ds = xr.open_dataset(filename)
        daily_ds = ds.resample(time="1D").mean()

        # Interpolate to the bathymetry mask grid
        interp_daily_ds = daily_ds.interp(
            longitude=mask.longitude, latitude=mask.latitude
        )

        # Apply the bathymetry mask to select the exact area
        masked_data = select(interp_daily_ds, mask)

        # Save in numpy format with shape (n_grid, f)
        for single_date in masked_data.time.values:
            date_str = pd.to_datetime(single_date).strftime("%Y%m%d")
            daily_data = masked_data.sel(time=single_date)
            combined_data = []
            for var in ds_variables:
                data = daily_data[var].values  # h, w
                combined_data.append(data)

            combined_data = np.stack(combined_data, axis=-1)  # h, w, f
            clean_data = np.nan_to_num(combined_data, nan=0.0)

            np.save(
                f"{path_prefix}/{date_str}.npy", clean_data[grid_mask, :]
            )  # n_grid, f
            print(f"Saved daily data to {path_prefix}/{date_str}.npy")

        # Increment to the next month
        if month == 7:
            current_date = current_date.replace(year=year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=month + 6, day=1)


def download_cerra(
    start_date,
    end_date,
    request_variables,
    ds_variables,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save daily CERRA data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    request_variables (list): List of variables to request from cds.
    ds_variables (list): List of variables in the dataset.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask.
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    client = cdsapi.Client()
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.strftime("%m")

        filename = f"{path_prefix}/{current_date.strftime('%Y%m')}.grib"

        client.retrieve(
            "reanalysis-cerra-single-levels",
            {
                "format": "grib",
                "data_type": "reanalysis",
                "level_type": "surface_or_atmosphere",
                "product_type": "forecast",
                "variable": request_variables,
                "year": year,
                "month": month,
                "day": list(range(1, 32)),
                "time": "00:00",
                "leadtime_hour": [
                    "1",
                    "3",
                    "6",
                    "9",
                    "12",
                    "15",
                    "18",
                    "21",
                ],
            },
            filename,
        )
        print(f"Downloaded {filename}")

        datasets = cfgrib.open_datasets(filename)
        updated_datasets = []
        for ds in datasets:
            ds = ds.drop_vars(["heightAboveGround"], errors="ignore")
            updated_datasets.append(ds)
        ds = xr.merge(updated_datasets)

        ds["u10"] = -ds.si10 * np.sin(np.deg2rad(ds.wdir10))
        ds["v10"] = -ds.si10 * np.cos(np.deg2rad(ds.wdir10))
        ds = ds.drop_vars(["si10", "wdir10"])

        # Average lead times for each day
        daily_ds = ds.resample(step="1D").mean()

        dims = daily_ds["longitude"].dims
        new_longitudes = np.where(
            daily_ds["longitude"] > 180,
            daily_ds["longitude"] - 360,
            daily_ds["longitude"],
        )
        daily_ds = daily_ds.assign_coords(longitude=(dims, new_longitudes))

        # Apply geographic filtering
        min_lon, max_lon = mask.longitude.min() - 1, mask.longitude.max() + 1
        min_lat, max_lat = mask.latitude.min() - 1, mask.latitude.max() + 1
        daily_ds = daily_ds.where(
            (daily_ds.longitude >= min_lon)
            & (daily_ds.longitude <= max_lon)
            & (daily_ds.latitude >= min_lat)
            & (daily_ds.latitude <= max_lat),
            drop=True,
        )

        # Set up the target grid
        target_lon, target_lat = np.meshgrid(mask.longitude, mask.latitude)
        target_points = np.vstack((target_lon.ravel(), target_lat.ravel())).T

        interp_dataset = xr.Dataset()
        all_variable_data = []
        variable_names = list(daily_ds.data_vars)

        # Collect flattened data for all variables across all time steps
        for var_name, da in daily_ds.data_vars.items():
            all_variable_data.append(da.values.reshape(len(daily_ds.time), -1))

        # Convert list to an array (time, npoints, nvars)
        all_variable_data = np.stack(all_variable_data, axis=-1)

        # Prepare the structure in interp_dataset for each variable
        for var_name in variable_names:
            interp_dataset[var_name] = xr.DataArray(
                np.empty(
                    (
                        len(daily_ds.time),
                        len(mask.latitude),
                        len(mask.longitude),
                    )
                ),
                coords={
                    "time": daily_ds.time,
                    "latitude": mask.latitude,
                    "longitude": mask.longitude,
                },
                dims=["time", "latitude", "longitude"],
            )

        # Interpolate each time step
        for time_idx in range(len(daily_ds.time)):
            lat_flat = daily_ds.latitude.values.flatten()
            lon_flat = daily_ds.longitude.values.flatten()
            coords = np.vstack((lon_flat, lat_flat)).T

            # Variables flattened for the current time step
            variable_flat = all_variable_data[
                time_idx, :, :
            ]  # (npoints, nvars)

            # Interpolate timestep
            interp = scipy.interpolate.LinearNDInterpolator(
                coords, variable_flat
            )
            interp_values = interp(target_points)  # (npoints, nvars)

            # Reshape interpolated values back to grid and assign to dataset
            for var_idx, var_name in enumerate(variable_names):
                interp_values_var = interp_values[:, var_idx].reshape(
                    target_lon.shape
                )
                interp_dataset[var_name][time_idx] = interp_values_var

        masked_data = select(interp_dataset, mask)

        # Save in numpy format with shape (n_grid, f)
        for single_date in masked_data.time.values:
            date_str = pd.to_datetime(single_date).strftime("%Y%m%d")
            daily_data = masked_data.sel(time=single_date)
            combined_data = []
            for var in ds_variables:
                data = daily_data[var].values  # h, w
                combined_data.append(data)

            combined_data = np.stack(combined_data, axis=-1)  # h, w, f
            clean_data = np.nan_to_num(combined_data, nan=0.0)

            np.save(
                f"{path_prefix}/{date_str}.npy", clean_data[grid_mask, :]
            )  # n_grid, f
            print(f"Saved daily data to {path_prefix}/{date_str}.npy")

        # Increment to the next month
        if month == "12":
            current_date = current_date.replace(year=year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=int(month) + 1, day=1)


def download_hres_forecast(
    start_date,
    start_time,
    static_path,
    path_prefix,
    request_variables,
    ds_variables,
    mask,
):
    """
    Download ECMWF HRES forecast data for the next 10 days starting from today.

    Args:
    start_date (datetime): The start date for data retrieval.
    start_time (int): The start time for data retrieval.
    static_path (str): Location of static data.
    path_prefix (str): Path where the files will be saved.
    request_variables (list): List of variables to download.
    ds_variables (list): List of variables in the dataset.
    mask (xarray.Dataset): Bathymetry mask
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    # Set up HRES client
    client = eo.Client(
        source="ecmwf",
        model="ifs",
        resol="0p25",
    )

    grib_filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.grib"

    # Retrieve data
    client.retrieve(
        date=start_date.strftime("%Y-%m-%d"),
        time=start_time,
        type="fc",
        stream="oper",
        step=list(range(0, 144, 3))
        + list(range(144, 241, 6)),  # 240 hours = 10 days
        param=request_variables,
        target=grib_filename,
    )
    print(f"Downloaded {grib_filename}")

    # Open the datasets and drop conflicting height
    datasets = []
    for var in request_variables:
        filter_keys = {"shortName": var}
        ds = xr.open_dataset(
            grib_filename, engine="cfgrib", filter_by_keys=filter_keys
        )  # t, h, w
        ds = ds.drop_vars(["heightAboveGround"], errors="ignore")
        datasets.append(ds)
    merged_ds = xr.merge(datasets)

    # Resample to daily averages
    daily_ds = merged_ds.resample(valid_time="1D").mean()

    # Interpolate onto the sea coordinates
    interp_daily_ds = daily_ds.interp(
        longitude=mask.longitude, latitude=mask.latitude
    )

    # Apply the bathymetry mask to select the exact area
    masked_data = select(interp_daily_ds, mask)

    # Stack as a numpy array
    combined_data = []
    for var in ds_variables:
        data = masked_data[var].values  # t, h, w
        combined_data.append(data)

    combined_data = np.stack(combined_data, axis=-1)  # t, h, w, f
    clean_data = np.nan_to_num(combined_data, nan=0.0)

    # Select only sea grid points
    sea_data = clean_data[:, grid_mask, :]  # t, n_grid, f

    filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.npy"
    np.save(filename, sea_data)
    print(f"Saved forecast data to {filename}")


def main():
    """
    Main function to organize the download and processing of oceanographic data.
    """
    parser = argparse.ArgumentParser(description="Download oceanographic data.")
    parser.add_argument(
        "-b",
        "--base_path",
        type=str,
        default="data/mediterranean/",
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default="1987-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default="2024-05-25",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        choices=["analysis", "reanalysis", "era5", "cerra"],
        help="Choose between analysis, reanalysis or era5",
    )
    parser.add_argument(
        "--static", action="store_true", help="Download static data"
    )
    parser.add_argument(
        "--forecast", action="store_true", help="Download today's forecast"
    )
    args = parser.parse_args()

    if args.forecast:
        start_date = datetime.today()
        end_date = start_date + timedelta(days=9)
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    static_path = args.base_path + "static/"
    raw_path = args.base_path + "raw/"
    reanalysis_path = raw_path + "reanalysis"
    analysis_path = raw_path + "analysis"
    cerra_path = raw_path + "cerra"
    era5_path = raw_path + "era5"
    hres00z_path = raw_path + "hres_00z"
    hres12z_path = raw_path + "hres_12z"
    forecast_path = raw_path + "forecast"
    bathymetry_mask_path = static_path + "bathy_mask.nc"

    os.makedirs(static_path, exist_ok=True)
    os.makedirs(reanalysis_path, exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(cerra_path, exist_ok=True)
    os.makedirs(era5_path, exist_ok=True)
    os.makedirs(hres00z_path, exist_ok=True)
    os.makedirs(hres12z_path, exist_ok=True)
    os.makedirs(forecast_path, exist_ok=True)

    mask = load_mask(bathymetry_mask_path)

    if args.static:
        download_static(static_path, mask)

    if args.data_source == "reanalysis":
        datasets = {
            "med-cmcc-cur-rean-d": ["uo", "vo"],
            "med-cmcc-mld-rean-d": ["mlotst"],
            "med-cmcc-sal-rean-d": ["so"],
            "med-cmcc-ssh-rean-d": ["zos"],
            "med-cmcc-tem-rean-d": ["thetao", "bottomT"],
        }
        version = "202012"

        # start_date = datetime(1987, 1, 1)
        # end_date = datetime(2022, 7, 31)

        download_data(
            start_date,
            end_date,
            datasets,
            version,
            static_path,
            reanalysis_path,
            mask,
        )

    if args.data_source == "analysis":
        datasets = {
            "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m": ["uo", "vo"],
            "cmems_mod_med_phy-mld_anfc_4.2km_P1D-m": ["mlotst"],
            "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m": ["so"],
            "cmems_mod_med_phy-ssh_anfc_4.2km_P1D-m": ["zos"],
            "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m": ["thetao", "bottomT"],
        }
        version = "202311"

        # start_date = datetime(2021, 11, 1)
        # end_date = datetime(2024, 5, 25)

        download_data(
            start_date,
            end_date,
            datasets,
            version,
            static_path,
            analysis_path,
            mask,
        )

    if args.data_source == "cerra":
        request_variables = [
            "10m_wind_direction",
            "10m_wind_speed",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_net_solar_radiation",
            "total_precipitation",
        ]
        ds_variables = ["u10", "v10", "t2m", "msl", "ssr", "tp"]
        download_cerra(
            start_date,
            end_date,
            request_variables,
            ds_variables,
            static_path,
            cerra_path,
            mask,
        )

    if args.data_source == "era5":
        request_variables = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "surface_net_solar_radiation",
            "total_precipitation",
        ]
        ds_variables = ["u10", "v10", "t2m", "msl", "ssr", "tp"]
        download_era5(
            start_date,
            end_date,
            request_variables,
            ds_variables,
            static_path,
            era5_path,
            mask,
        )

    if args.forecast:
        datasets = {
            "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m": ["uo", "vo"],
            "cmems_mod_med_phy-mld_anfc_4.2km_P1D-m": ["mlotst"],
            "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m": ["so"],
            "cmems_mod_med_phy-ssh_anfc_4.2km_P1D-m": ["zos"],
            "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m": ["thetao", "bottomT"],
        }
        version = "202311"

        download_forecast(
            start_date,
            end_date,
            datasets,
            version,
            static_path,
            forecast_path,
            mask,
        )

        request_variables = ["10u", "10v", "2t", "msl", "ssr", "tp"]
        ds_variables = ["u10", "v10", "t2m", "msl", "ssr", "tp"]

        for start_time, hres_path in zip([0, 12], [hres00z_path, hres12z_path]):
            download_hres_forecast(
                start_date,
                start_time,
                static_path,
                hres_path,
                request_variables,
                ds_variables,
                mask,
            )


if __name__ == "__main__":
    main()
