import numpy as np
import os
import argparse
from glob import glob


def prepare_states(in_directory, out_directory, n_states, prefix):
    # Get all .npy files sorted by date assuming filenames are date-based
    files = sorted(glob(os.path.join(in_directory, '*.npy')))

    # Ensure output directory exists
    os.makedirs(out_directory, exist_ok=True)

    # Process each file, concatenate with the next t-1 files
    for i in range(len(files) - n_states + 1):
        out_filename = f'{prefix}_{os.path.basename(files[i])}'
        out_file = os.path.join(out_directory, out_filename)
        
        if os.path.isfile(out_file):
            continue
        
        state_sequence = []

        # Load each state to concatenate
        for j in range(n_states):
            state_sequence.append(np.load(files[i + j]))

        # Concatenate along new axis (time axis)
        full_state = np.stack(state_sequence, axis=0)
        
        # Save concatenated data to the output directory using the first file's basename
        np.save(out_file, full_state)
        print(f"Saved concatenated file: {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare state sequences from Baltic Sea data files.')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='Output directory for concatenated files')
    parser.add_argument('-n', '--n_states', type=int, default=8, help='Number of states to concatenate')
    parser.add_argument('-p', '--prefix', type=str, required="RE", help='Prefix for the output files')
    args = parser.parse_args()

    prepare_states(args.data_dir, args.out_dir, args.n_states, args.prefix)


if __name__ == '__main__':
    main()
