import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import argparse
import sys, os

COLUMN_FILTERS = [
    'train reward of=team-0',
    'train top reward of=team-0',
    'train top-rank ratio of=team-0',
    'train top-rank ratio of=team-1',
    'train draw ratio',
    'Mean Reward',
    'Top Reward',
    'Top-Rank Ratio'
]

COLUMN_RENAME_MAP = {
    # 'train reward of=team-0': 'Mean Reward',
    # 'train top reward of=team-0': 'Top Reward',
    # 'train top-rank ratio of=team-0': 'Top-Rank Ratio'
}

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert 'Time' in df.columns, "Column 'Time' not found"

    selected_cols = ['Time']
    for filter_pattern in COLUMN_FILTERS:
        matched = [col for col in df.columns if filter_pattern in col]
        if matched:
            selected_cols.extend(matched)

    return df[selected_cols]

def merge_csv_files(csv_files: List[str], output_file: str, visualize: bool = True):
    if len(csv_files) < 2:
        print("Need at least 2 CSV files to merge")
        sys.exit(1)

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_filtered = filter_columns(df)
        dfs.append(df_filtered)

    headers = dfs[0].columns.tolist()
    for i, df in enumerate(dfs[1:], 1):
        assert df.columns.tolist() == headers, f"Headers mismatch at file {i}: {csv_files[i]}"

    assert headers[0] == 'Time', f"First column must be 'Time', got '{headers[0]}'"

    merged_dfs = [dfs[0]]
    time_offset = dfs[0]['Time'].max()

    for df in dfs[1:]:
        df_copy = df.copy()
        df_copy['Time'] = df_copy['Time'] + time_offset
        merged_dfs.append(df_copy)
        time_offset = df_copy['Time'].max()

    result = pd.concat(merged_dfs, ignore_index=True)
    result.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    result.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files into {output_file}")
    print(f"Total rows: {len(result)}, Time range: {result['Time'].min():.2f} - {result['Time'].max():.2f}")

    if visualize:
        plot_results(result, output_file.replace('.csv', '.png'))

def plot_results(df: pd.DataFrame, output_file: str):
    metric_cols = [col for col in df.columns if col != 'Time']
    n_metrics = len(metric_cols)

    if n_metrics == 0:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        ax.plot(df['Time'], df[col], linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple CSV files along Time axis')
    parser.add_argument('csv_files', nargs='*', help='CSV files to merge')
    parser.add_argument('-o', '--output', default='merged.csv', help='Output file name')
    parser.add_argument('-d', '--directory', default=None, help='Auto directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable visualization')

    args = parser.parse_args()
    if args.directory is not None:
        if not os.path.exists(args.directory):
            print(f"Directory {args.directory} does not exist")
            sys.exit(1)
        NEEDED = [
            "rec copy.csv",
            "rec copy 2.csv",
            "rec copy 3.csv",
            "rec copy 4.csv",
            "rec copy 5.csv",
            "rec copy 6.csv",
            "rec copy 7.csv",
            "rec copy 8.csv",
            "rec copy 9.csv",
            "rec copy 10.csv",
            "rec.csv",
        ]
        csv_files = []
        output_file = os.path.join(args.directory, "merged.csv")
        files = os.listdir(args.directory)
        files = [file for file in files if file.endswith('.csv')]
        rec_files = [file for file in files if file.startswith("rec")]
        for needed in NEEDED:
            if not needed in rec_files:
                print(f"{needed} not found in {args.directory}: {rec_files}")
            else:
                csv_files.append(os.path.join(args.directory, needed))
    else:
        if len(args.csv_files) < 2:
            print("Need at least 2 CSV files to merge")
            sys.exit(1)
        csv_files = args.csv_files
        output_file = args.output
    
    merge_csv_files(csv_files=csv_files, output_file=output_file, visualize=not args.no_plot)
