# Standard library
import argparse
import os

# Third-party
import copernicusmarine as cm


def prepare_sst(input_dir, output_dir):
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


def main():
    parser = argparse.ArgumentParser(description="Prepare observation data.")
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        choices=["sst"],
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


if __name__ == "__main__":
    main()
