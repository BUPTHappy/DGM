import os
import sys
import subprocess
import time
import multiprocessing as mp
import click

PROJECT_NAME = "uva"

# UMI Cup Arrangement datasets
# cup_arrangement_1 (lab) is smaller and faster for quick experiments
# cup_arrangement_0 (wild) is larger with more diverse real-world scenarios
ALL_DATASETS = {
    "cup_arrangement_0": {
        "url": "https://real.stanford.edu/umi/data/cup_in_the_wild/cup_in_the_wild.zarr.zip",
        "name": "cup_in_the_wild",
        "description": "Larger dataset with diverse real-world scenarios"
    },
    "cup_arrangement_1": {
        "url": "https://real.stanford.edu/umi/data/cup_arrangement/cup_in_the_lab.zarr.zip",
        "name": "cup_in_the_lab",
        "description": "Smaller dataset, faster for quick experiments (recommended for fast training)"
    },
}

# Default: only download lab dataset (smaller, faster)
DATASETS = {
    "cup_arrangement_1": ALL_DATASETS["cup_arrangement_1"]["url"]
}


def download_data(dataset_name: str, url: str, output_dir: str) -> None:
    """
    Download the data from the given URL and save it to the given dataset name.
    """
    os.makedirs(output_dir, exist_ok=True)
    shm_data_dir = f"/dev/shm/{PROJECT_NAME}/temp"
    if ";" in url:
        urls = url.split(";")
        os.makedirs(shm_data_dir, exist_ok=True)

        def download_url(url: str, id: int, output_dir: str) -> None:
            if os.path.exists(f"{output_dir}/{dataset_name}_part_{id}"):
                print(
                    f"Skipping downloading {dataset_name} because {output_dir}/{dataset_name}_part_{id} already exists"
                )
            else:
                print(
                    f"Downloading {dataset_name} from {url} to {output_dir}/{dataset_name}_part_{id}"
                )
                subprocess.run(
                    ["wget", url, "-O", f"{output_dir}/{dataset_name}_part_{id}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            print(
                f"Moving {output_dir}/{dataset_name}_part_{id} to {shm_data_dir}/{dataset_name}_part_{id}"
            )
            subprocess.run(
                ["mv", f"{output_dir}/{dataset_name}_part_{id}", shm_data_dir],
                check=True,
            )

        for i, url in enumerate(urls):
            download_url(url, i, output_dir)

        print(f"Merging {dataset_name}.zarr.zip")
        subprocess.run(
            [
                "cat",
                f"{shm_data_dir}/{dataset_name}_part_*",
                ">",
                f"{shm_data_dir}/{dataset_name}.zarr.zip",
            ],
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(
            f"Moving {shm_data_dir}/{dataset_name}.zarr.zip to {output_dir}/{dataset_name}.zarr.zip"
        )
        subprocess.run(
            ["mv", f"{shm_data_dir}/{dataset_name}.zarr.zip", output_dir], check=True
        )
        subprocess.run(["rm", "-rf", shm_data_dir], check=True)

    else:
        print(
            f"Downloading {dataset_name} from {url} to {output_dir}/{dataset_name}.zarr.zip"
        )
        subprocess.run(
            ["wget", url, "-O", f"{output_dir}/{dataset_name}.zarr.zip"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"Downloaded {dataset_name} to {output_dir}/{dataset_name}.zarr.zip")


def convert_zip_to_lz4(dataset_name: str, data_dir: str):
    # First copy zip file to shared memory /dev/shm
    shm_data_dir = f"/dev/shm/{PROJECT_NAME}/temp"
    os.makedirs(shm_data_dir, exist_ok=True)
    shm_file = f"{shm_data_dir}/{dataset_name}.zarr.zip"
    zip_file = f"{data_dir}/{dataset_name}.zarr.zip"

    print(f"Copying {zip_file} to {shm_file}")
    subprocess.run(["cp", zip_file, shm_file], check=True)

    print(f"Unzipping {shm_file} to {shm_data_dir}/{dataset_name}.zarr")
    subprocess.run(
        ["unzip", shm_file, "-d", f"{shm_data_dir}/{dataset_name}.zarr"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"Removing {shm_file}")
    subprocess.run(["rm", shm_file], check=True)

    print(f"Compressing {dataset_name}.zarr to {dataset_name}.zarr.tar.lz4")
    subprocess.run(
        [f"tar cf - {dataset_name}.zarr | lz4 -c > {dataset_name}.zarr.tar.lz4"],
        cwd=shm_data_dir,
        shell=True,
        check=True,
    )

    zip_file_dir = os.path.dirname(zip_file)
    if zip_file_dir.endswith("zip"):
        # data_dir/zip
        # data_dir/lz4
        zip_file_dir = os.path.dirname(zip_file_dir)
    os.makedirs(f"{zip_file_dir}/lz4", exist_ok=True)
    print(f"Copying {shm_data_dir}/{dataset_name}.zarr.tar.lz4 to {zip_file_dir}/lz4")
    subprocess.run(
        ["cp", f"{shm_data_dir}/{dataset_name}.zarr.tar.lz4", f"{zip_file_dir}/lz4"],
        check=True,
    )

    print(f"Removing {shm_data_dir}/{dataset_name}.zarr")
    subprocess.run(["rm", "-rf", f"{shm_data_dir}/{dataset_name}.zarr"], check=True)

    print(f"Removing {shm_data_dir}/{dataset_name}.zarr.tar.lz4")
    subprocess.run(["rm", f"{shm_data_dir}/{dataset_name}.zarr.tar.lz4"], check=True)


def process_dataset(dataset_name: str, dataset_url: str, data_dir: str, extract: bool = True) -> None:
    """
    Download and optionally extract dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_url: URL to download from
        data_dir: Directory to save data
        extract: Whether to extract the zip file (default: True)
    """
    zip_file = f"{data_dir}/{dataset_name}.zarr.zip"
    extracted_dir = f"{data_dir}/{dataset_name}.zarr"
    
    # Download if not exists
    if not os.path.exists(zip_file) and not os.path.exists(extracted_dir):
        download_data(dataset_name, dataset_url, data_dir)
    elif os.path.exists(zip_file):
        print(f"Zip file already exists: {zip_file}")
    elif os.path.exists(extracted_dir):
        print(f"Extracted directory already exists: {extracted_dir}")
        return
    
    # Extract if requested and not already extracted
    if extract and os.path.exists(zip_file) and not os.path.exists(extracted_dir):
        print(f"Extracting {zip_file}...")
        # Create a temporary extraction directory
        temp_extract_dir = f"{data_dir}/_temp_{dataset_name}"
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        try:
            subprocess.run(
                ["unzip", "-q", zip_file, "-d", temp_extract_dir],
                check=True,
            )
            
            # Find the actual zarr directory inside (usually there's one level of nesting)
            extracted_contents = os.listdir(temp_extract_dir)
            if len(extracted_contents) == 1:
                # If there's a single directory, it's likely the zarr directory
                nested_path = os.path.join(temp_extract_dir, extracted_contents[0])
                if os.path.isdir(nested_path):
                    # Check if it looks like a zarr directory (has .zarray or .zgroup files)
                    if any(f.endswith('.zarray') or f.endswith('.zgroup') for f in os.listdir(nested_path) if os.path.isfile(os.path.join(nested_path, f))):
                        # This is the zarr directory, move it to the final location
                        os.rename(nested_path, extracted_dir)
                    else:
                        # It's a container directory, move its contents
                        os.makedirs(extracted_dir, exist_ok=True)
                        for item in os.listdir(nested_path):
                            os.rename(
                                os.path.join(nested_path, item),
                                os.path.join(extracted_dir, item)
                            )
                else:
                    # It's a file, just move it
                    os.rename(nested_path, extracted_dir)
            else:
                # Multiple items, move all to extracted_dir
                os.makedirs(extracted_dir, exist_ok=True)
                for item in extracted_contents:
                    os.rename(
                        os.path.join(temp_extract_dir, item),
                        os.path.join(extracted_dir, item)
                    )
            
            print(f"✓ Extracted to {extracted_dir}")
        finally:
            # Clean up temp directory
            if os.path.exists(temp_extract_dir):
                subprocess.run(["rm", "-rf", temp_extract_dir], check=False)


@click.command()
@click.option("--data_dir", type=str, default="data", help="Directory to save datasets (default: data)")
@click.option("--extract/--no-extract", default=True, help="Extract zip files after downloading (default: True)")
@click.option("--parallel/--no-parallel", default=True, help="Download datasets in parallel (default: True)")
@click.option("--dataset", type=click.Choice(["lab", "wild", "both"], case_sensitive=False), 
              default="lab", help="Which dataset to download: lab (smaller, faster), wild (larger), or both (default: lab)")
def main(data_dir: str, extract: bool, parallel: bool, dataset: str):
    """
    Download UMI cup arrangement datasets.
    
    Options:
    - lab: cup_in_the_lab (smaller, faster for quick experiments) [RECOMMENDED]
    - wild: cup_in_the_wild (larger, more diverse)
    - both: download both datasets
    """
    # Select datasets based on choice
    selected_datasets = {}
    if dataset.lower() == "lab" or dataset.lower() == "both":
        selected_datasets["cup_arrangement_1"] = ALL_DATASETS["cup_arrangement_1"]["url"]
    if dataset.lower() == "wild" or dataset.lower() == "both":
        selected_datasets["cup_arrangement_0"] = ALL_DATASETS["cup_arrangement_0"]["url"]
    
    if not selected_datasets:
        print("Error: No datasets selected")
        return
    
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading {len(selected_datasets)} cup arrangement dataset(s) to {data_dir}")
    print("=" * 60)
    for name, info in ALL_DATASETS.items():
        if name in selected_datasets:
            print(f"  - {name}: {info['name']} ({info['description']})")
    print("=" * 60)
    
    if parallel and len(selected_datasets) > 1:
        num_processes = min(mp.cpu_count(), len(selected_datasets))
        print(f"Using {num_processes} parallel processes")
        with mp.Pool(num_processes) as pool:
            pool.starmap(
                process_dataset,
                [(dataset_name, url, data_dir, extract) for dataset_name, url in selected_datasets.items()],
            )
    else:
        print("Downloading sequentially")
        for dataset_name, url in selected_datasets.items():
            process_dataset(dataset_name, url, data_dir, extract)
    
    print("=" * 60)
    print("Download complete!")
    print(f"\nDatasets saved to: {data_dir}")
    if extract:
        print("\nExtracted zarr directories:")
        for dataset_name in selected_datasets.keys():
            zarr_dir = f"{data_dir}/{dataset_name}.zarr"
            if os.path.exists(zarr_dir):
                print(f"  - {zarr_dir}")
                # Show which dataset this is
                info = ALL_DATASETS.get(dataset_name, {})
                if info:
                    print(f"    ({info.get('name', '')})")
    else:
        print("\nZip files:")
        for dataset_name in selected_datasets.keys():
            zip_file = f"{data_dir}/{dataset_name}.zarr.zip"
            if os.path.exists(zip_file):
                print(f"  - {zip_file}")
                info = ALL_DATASETS.get(dataset_name, {})
                if info:
                    print(f"    ({info.get('name', '')})")


if __name__ == "__main__":
    main()
