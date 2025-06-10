from datasets import load_dataset
from datasets import Dataset
from pathlib import Path
from itertools import islice
import argparse
import torch

def download_dev_split(data_dir: Path, max_samples = 1000):
    dataset = iter(load_dataset("Skylion007/openwebtext", split="train", streaming=True))

    partial_data = []
    for _ in range(max_samples): 
        partial_data.append(next(dataset))

    partial_dataset = Dataset.from_list(partial_data)
    partial_dataset.save_to_disk(data_dir / "openwebtext")

def download_openwebtext(data_dir: Path, dev=False):
    print("Downloading OpenWebText dataset...")
    if (dev):
        download_dev_split(data_dir)
    else: 
        openwebtext = load_dataset("Skylion007/openwebtext", split="train")
        openwebtext.save_to_disk(data_dir / "openwebtext") # type: ignore
        print("OpenWebText dataset downloaded successfully!")
    
def download_glue_datasets(data_dir: Path): 
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
    parser.add_argument("--glue", action="store_true", help="Download a subset of the dataset for development")
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    download_openwebtext(data_dir, args.dev) 
    if (args.glue): 
        download_glue_datasets(data_dir)
