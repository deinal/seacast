# Standard library
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

# Third-party
import copernicusmarine as cm
import numpy as np


def download_forecast_data(
    dataset_phy, dataset_bio, vars_phy, vars_bio, version, depth, path_prefix
):
    """
    Downloads forecast data for specified physical and biological variables.

    Args:
        dataset_phy (str): ID of the physical dataset to be fetched.
        dataset_bio (str): ID of the biological dataset to be fetched.
        vars_phy (list): List of physical variables to download.
        vars_bio (list): List of biological variables to download.
        version (str): Dataset version.
        depth (float): Specific depth from which to retrieve data.
        path_prefix (str): Directory path where the numpy file will be saved.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(
        days=2
    )  # Include yesterday and the day before
    end_date = today + timedelta(days=5)  # Include a six day forecast

    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_date_str = end_date.strftime("%Y-%m-%dT00:00:00")

    print(f"Downloading data from {start_date_str} to {end_date_str}")

    # Fetch physical data for 8 days
    phy_data = cm.open_dataset(
        dataset_id=dataset_phy,
        dataset_version=version,
        service="arco-geo-series",
        variables=vars_phy,
        start_datetime=start_date_str,
        end_datetime=end_date_str,
        minimum_depth=depth,
        maximum_depth=depth,
    )

    # Fetch biological data for 8 days
    bio_data = cm.open_dataset(
        dataset_id=dataset_bio,
        dataset_version=version,
        service="arco-geo-series",
        variables=vars_bio,
        start_datetime=start_date_str,
        end_datetime=end_date_str,
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
    combined_data = np.stack(all_data, axis=-1)  # (T, W, H, F)
    clean_data = np.nan_to_num(combined_data, nan=0.0)

    # Save combined data as one numpy file
    filename = f"{path_prefix}/{today.strftime('%Y%m%d')}.npy"
    np.save(filename, clean_data)
    print(f"Saved combined data to {filename}")


def main():
    """
    Main execution function that triggers the download of forecast data.
    """
    parser = ArgumentParser(description="Forecast download arguments")
    parser.add_argument(
        "--dir", type=str, help="Where to store downloaded data"
    )
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

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

    download_forecast_data(
        "cmems_mod_bal_phy_anfc_P1D-m",
        "cmems_mod_bal_bgc_anfc_P1D-m",
        vars_phy,
        vars_bio,
        dataset_version,
        depth,
        args.dir,
    )


if __name__ == "__main__":
    main()
