from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
from itertools import islice
import argparse
import torch

def download_dev_split(data_dir: Path, max_samples = 1000):
    dataset = load_dataset("openwebtext", streaming=True)

    partial_data = [{"text": x} for x in islice(dataset, max_samples)]
    partial_dataset = Dataset.from_list(partial_data)
    partial_dataset.save_to_disk(data_dir / "openwebtext")

def download_datasets(dev=False):
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download OpenWebText dataset for training
    print("Downloading OpenWebText dataset...")
    if (dev):
        download_dev_split(data_dir)
    else: 
        openwebtext = load_dataset("openwebtext", split="train")
        openwebtext.save_to_disk(data_dir / "openwebtext") # type: ignore
        print("OpenWebText dataset downloaded successfully!")
    
    # Download GLUE datasets for evaluation
    print("\nDownloading GLUE datasets...")
    glue_tasks = [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "stsb",
        "mnli",
        "qnli",
        "rte",
        "wnli"
    ]
    splits = ["train", "validation", "test"]
    for task in glue_tasks:
        print(f"Downloading {task}...")
        try:
            task_dir = data_dir / f"glue_{task}"
            task_dir.mkdir(parents=True, exist_ok=True)
            for split in splits:
                try:
                    dataset = load_dataset("glue", task, split=split)
                    dataset.save_to_disk(task_dir / split) # type: ignore
                    print(f"  {split} split downloaded!")
                except Exception as e:
                    print(f"  {split} split not available: {str(e)}")
            print(f"{task} done!\n")
        except Exception as e:
            print(f"Error downloading {task}: {str(e)}")
    print("\nAll datasets have been downloaded to the 'data' directory!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dev", action="store_true", help="Download a subset of the dataset for development")
    args = parser.parse_args()
    download_datasets(args.dev) 
