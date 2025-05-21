from datasets import load_dataset
import os
import torch
from itertools import islice

dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

max_samples = 1000
n_tensors_per_file = 50
seed = 42
save_path = "./data"

os.makedirs(save_path, exist_ok=True)

buffer = []
file_count = 0

for i, example in enumerate(islice(dataset, max_samples)):
    buffer.append(example)

    if len(buffer) >= n_tensors_per_file:
        file_name = os.path.join(save_path, f"chunk_{file_count:04d}.pt")
        torch.save(buffer, file_name)
        buffer.clear()
        file_count += 1

if buffer:
    file_name = os.path.join(save_path, f"chunk_{file_count:04d}.pt")
    torch.save(buffer, file_name)

