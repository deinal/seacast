# Standard library
import os
from datetime import datetime, timedelta

# Third-party
import copernicusmarine as cm
import numpy as np


def download_grid(path_prefix):
    """
    Download and save ocean depth and land mask data as numpy files.

    Args:
    path_prefix (str): The directory path prefix where the files will be saved.
    """
    grid = cm.open_dataset(
        dataset_id="cmems_mod_bal_phy_my_static",
        dataset_part="bathy",
        dataset_version="202303",
        service="static-arco",
        variables=["deptho", "mask"],
        minimum_depth=0.5,
        maximum_depth=0.5,
    )

    ocean_depth = np.nan_to_num(grid.deptho.squeeze(), nan=0.0)
    np.save(f"{path_prefix}/ocean_depth.npy", ocean_depth)

    land_mask = ~(grid.mask.astype(bool)).squeeze()
    np.save(f"{path_prefix}/land_mask.npy", land_mask)

    y_indices, x_indices = np.indices(land_mask.shape)
    nwp_xy = np.array([x_indices, y_indices])
    np.save(f"{path_prefix}/nwp_xy.npy", nwp_xy)


def download_data(
    start_date,
    end_date,
    dataset_phy,
    dataset_bio,
    vars_phy,
    vars_bio,
    version,
    depth,
    path_prefix,
):
    """
    Download and save daily physics and biogeochemistry data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    dataset_phy (str): Dataset ID for the physical data.
    dataset_bio (str): Dataset ID for the biogeochemical data.
    vars_phy (list): Variables to retrieve from the physical dataset.
    vars_bio (list): Variables to retrieve from the biogeochemical dataset.
    version (str): Dataset version.
    depth (float): Depth at which to retrieve the data.
    path_prefix (str): The directory path prefix where the files will be saved.
    """

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%dT00:00:00")

        # Open ocean physics dataset
        phy_data = cm.open_dataset(
            dataset_id=dataset_phy,
            dataset_version=version,
            dataset_part="default",
            service="arco-geo-series",
            variables=vars_phy,
            start_datetime=date_str,
            end_datetime=date_str,
            minimum_depth=depth,
            maximum_depth=depth,
        )

        # Open biogeochemistry dataset
        bio_data = cm.open_dataset(
            dataset_id=dataset_bio,
            dataset_version=version,
            dataset_part="default",
            service="arco-geo-series",
            variables=vars_bio,
            start_datetime=date_str,
            end_datetime=date_str,
            minimum_depth=depth,
            maximum_depth=depth,
        )

        # Combine data arrays
        all_data = []
        for var in vars_phy:
            data = np.squeeze(phy_data[var].values)
            all_data.append(data)

        for var in vars_bio:
            data = np.squeeze(bio_data[var].values)
            all_data.append(data)

        # Close datasets
        phy_data.close()
        bio_data.close()

        # Concatenate all data along a new axis
        combined_data = np.stack(all_data, axis=-1)
        clean_data = np.nan_to_num(combined_data, nan=0.0)

        # Save combined data as one numpy file
        filename = f"{path_prefix}/{current_date.strftime('%Y%m%d')}.npy"
        np.save(filename, clean_data)
        print(filename)

        # Next day
        current_date += timedelta(days=1)


def main():
    """
    Main function to organize the download and processing of oceanographic data.
    """
    path_prefix_static = "data/baltic_sea/static"
    path_prefix_reanalysis = "data/baltic_sea/raw/reanalysis"
    path_prefix_analysis = "data/baltic_sea/raw/analysis"

    os.makedirs(path_prefix_static, exist_ok=True)
    os.makedirs(path_prefix_analysis, exist_ok=True)
    os.makedirs(path_prefix_reanalysis, exist_ok=True)

    download_grid(path_prefix_static)

    # Reanalysis
    vars_phy = [
        "thetao",
        "bottomT",
        "mlotst",
        "siconc",
        "sithick",
        "so",
        "sob",
        "uo",
        "vo",
    ]
    vars_bio = ["chl", "nh4", "no3", "o2", "o2b", "ph", "po4", "spco2", "zsd"]
    dataset_version = "202303"
    depth = 0.5
    start_date = datetime(1993, 1, 1)
    end_date = datetime(2021, 12, 31)
    download_data(
        start_date,
        end_date,
        "cmems_mod_bal_phy_my_P1D-m",
        "cmems_mod_bal_bgc_my_P1D-m",
        vars_phy,
        vars_bio,
        dataset_version,
        depth,
        path_prefix_reanalysis,
    )

    # Analysis
    vars_phy = [
        "thetao",
        "bottomT",
        "mlotst",
        "siconc",
        "sithick",
        "so",
        "sob",
        "uo",
        "vo",
    ]
    vars_bio = ["chl", "nh4", "no3", "o2", "o2b", "pH", "po4", "spCO2", "zsd"]
    dataset_version = "202311"
    depth = 0.5
    start_date = datetime(2021, 11, 1)
    end_date = datetime(2024, 4, 30)
    download_data(
        start_date,
        end_date,
        "cmems_mod_bal_phy_anfc_P1D-m",
        "cmems_mod_bal_bgc_anfc_P1D-m",
        vars_phy,
        vars_bio,
        dataset_version,
        depth,
        path_prefix_analysis,
    )


if __name__ == "__main__":
    main()
