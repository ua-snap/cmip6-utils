import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmip6_dir",
        type=str,
        help="Path to base cmip6 directory",
        required=True,
    )
    args = parser.parse_args()
    return Path(args.cmip6_dir)


def remove_old_versions(cmip6_dir):

    # Create a dictionary to store file names and their corresponding file paths
    # Iterate through all file paths and populate the dictionary
    all_fps = cmip6_dir.glob("**/*.nc")
    file_dict = defaultdict(list)
    for fp in all_fps:
        file_dict[fp.name].append(fp)

    # Find file names with more than one file path (duplicates)
    duplicate_files = {
        file_name: paths for file_name, paths in file_dict.items() if len(paths) > 1
    }

    remove_paths = []
    for file_name, paths in duplicate_files.items():
        # Extract the version date from the parent directory name, all should be format vYYYYMMDD
        paths.sort(
            key=lambda p: datetime.strptime(p.parent.name.replace("v", ""), "%Y%m%d")
        )
        # Add all but the last path to the list of paths to remove
        remove_paths.extend(paths[:-1])

    # Ask for user confirmation before removing files
    if len(remove_paths) > 0:
        print("The following files will be removed:")
        if len(remove_paths) > 20:
            for i in range(0, len(remove_paths), 20):
                print("\n".join(str(path) for path in remove_paths[i : i + 20]))
                if i + 20 < len(remove_paths):
                    if input("Show next 20 files? (enter == yes): ").strip():
                        break
        else:
            for path in remove_paths:
                print(path)

    else:
        print("No files to remove.")
        return

    confirm = input("Do you want to proceed? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Operation cancelled.")
        return

    # Remove the old versions
    for path in remove_paths:
        path.unlink()

    print(f"Old versions removed ({len(remove_paths)} removed).")


if __name__ == "__main__":
    cmip6_directory = parse_args()
    remove_old_versions(cmip6_directory)
