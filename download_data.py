# Standard library
import argparse
import os
from datetime import datetime, timedelta

# Third-party
import copernicusmarine as cm
import numpy as np
import xarray as xr

# First-party
from neural_lam import constants


def load_mask(path):
    """
    Load bathymetry mask.

    Args:
    path (str): Path to the bathymetry mask file.

    Returns:
    mask (xarray.DataArray): Bathymetry mask.
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
    dataset (xarray.DataArray): Input dataset.
    mask (xarray.DataArray): Bathymetry mask.

    Returns:
    mask (xarray.DataArray): Masked dataset.
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
    mask (xarray.DataArray): Bathymetry mask.
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

    sea_mask = ~np.isnan(bathy_data.mask)
    np.save(f"{path_prefix}/sea_mask.npy", sea_mask)

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
    path_prefix_static,
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
    path_prefix_static (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.DataArray): Bathymetry mask
    """

    np_mask = np.load(f"{path_prefix_static}/sea_mask.npy")[0]

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%dT00:00:00")
        filename = f"{path_prefix}/{current_date.strftime('%Y%m%d')}.npy"

        if os.path.isfile(filename):
            current_date += timedelta(days=1)
            continue

        all_data = []

        for dataset_id, variables in datasets.items():

            # Load ocean physics dataset
            dataset = cm.open_dataset(
                dataset_id=dataset_id,
                dataset_version=version,
                dataset_part="default",
                service="arco-geo-series",
                variables=variables,
                start_datetime=date_str,
                end_datetime=date_str,
                minimum_depth=constants.DEPTHS[0],
                maximum_depth=constants.DEPTHS[-1],
            )

            dataset = select(dataset, mask)
            for var in variables:
                data = dataset[var].isel(time=0).values
                if var == "bottomT":
                    data = data[:, :, 0]
                if len(data.shape) == 2:
                    data = data[np.newaxis, ...]
                data = data.transpose(1, 2, 0)  # h, w, f
                all_data.append(data)

            dataset.close()

        # Concatenate all data along a new axis
        combined_data = np.concatenate(all_data, axis=-1)
        clean_data = np.nan_to_num(combined_data, nan=0.0)

        # Select only sea grid points
        sea_data = clean_data[np_mask, :]  # n_grid, f

        # Save combined data as one numpy file
        np.save(filename, sea_data)
        print(filename)

        # Next day
        current_date += timedelta(days=1)


def download_forecast(
    start_date,
    end_date,
    datasets,
    version,
    path_prefix_static,
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
    path_prefix_static (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.DataArray): Bathymetry mask
    """
    np_mask = np.load(f"{path_prefix_static}/sea_mask.npy")[0]

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
            print(var, data.shape)
            all_data.append(data)

        dataset.close()

    # Concatenate all data along the feature dimension
    combined_data = np.concatenate(all_data, axis=-1)
    combined_data = np.nan_to_num(combined_data, nan=0.0)

    # Select only sea grid points
    sea_data = combined_data[:, np_mask, :]  # t, n_grid, f

    # Save combined data as one numpy file
    np.save(filename, sea_data)
    print(f"Saved forecast data to {filename}")


def main():
    """
    Main function to organize the download and processing of oceanographic data.
    """
    parser = argparse.ArgumentParser(description="Download oceanographic data.")
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
        default="2022-07-31",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        choices=["analysis", "reanalysis"],
        help="Choose between analysis or reanalysis",
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

    path_prefix_static = "data/mediterranean/static"
    path_prefix_reanalysis = "data/mediterranean/raw/reanalysis"
    path_prefix_analysis = "data/mediterranean/raw/analysis"
    path_prefix_forecast = "data/mediterranean/raw/forecast"
    bathymetry_mask_path = "data/mediterranean/static/bathy_mask.nc"

    os.makedirs(path_prefix_static, exist_ok=True)
    os.makedirs(path_prefix_reanalysis, exist_ok=True)
    os.makedirs(path_prefix_analysis, exist_ok=True)
    os.makedirs(path_prefix_forecast, exist_ok=True)

    mask = load_mask(bathymetry_mask_path)

    if args.static:
        download_static(path_prefix_static, mask)

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
            path_prefix_static,
            path_prefix_reanalysis,
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
            path_prefix_static,
            path_prefix_analysis,
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
            path_prefix_static,
            path_prefix_forecast,
            mask,
        )


if __name__ == "__main__":
    main()
