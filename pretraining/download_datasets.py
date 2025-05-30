from datasets import load_dataset
import os
from pathlib import Path

def download_datasets():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download OpenWebText dataset for training
    print("Downloading OpenWebText dataset...")
    openwebtext = load_dataset("openwebtext", split="train")
    openwebtext.save_to_disk(data_dir / "openwebtext")
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
                    dataset.save_to_disk(task_dir / split)
                    print(f"  {split} split downloaded!")
                except Exception as e:
                    print(f"  {split} split not available: {str(e)}")
            print(f"{task} done!\n")
        except Exception as e:
            print(f"Error downloading {task}: {str(e)}")
    print("\nAll datasets have been downloaded to the 'data' directory!")

if __name__ == "__main__":
    download_datasets() 