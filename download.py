from datasets import load_dataset
import os

dataset_path = "./data/"
dataset = None

dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
subset = []
for i, sample in enumerate(dataset):
    subset.append(sample)
    if i >= 10:
        break

print(subset)

#
# if (not os.path.exists(dataset_path)):
#     dataset = load_dataset("Skylion007/openwebtext", streaming=True)
#     os.mkdir(dataset_path)
#     dataset.save_to_disk(dataset_path)
# else: 
#     dataset = load_dataset(dataset_path)
