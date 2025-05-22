import os
import numpy as np
import torch 
import multiprocessing
from itertools import batched
from functools import partial


def run_processes(tokenizer, processes, n_tensors_per_file):
    src_dir = "./data/"
    dest_dir = "./data_preprocessed/"

    os.makedirs(dest_dir, exist_ok=True)

    files = os.listdir(src_dir)
    batched_files = list(batched(files, processes))
    print(len(batched_files))

    partial_func = partial(tokenize_process, tokenizer, src_dir)

    pool = multiprocessing.Pool(processes)
    features = pool.map(partial_func, batched_files)


    temp = []
    counter = 0
    for feature in features: 
        temp.append(feature)
        if (len(temp) >= n_tensors_per_file): 
            torch.save(temp, dest_dir + f"/file_{counter}.pt")
            counter += 1
            temp.clear()


def tokenize_process(tokenizer, src_dir, files):
    features = []
    for file in files: 
        loaded = torch.load(src_dir + "/" + file)
        for entry in loaded: 
            tokenized = tokenizer(entry["text"])["input_ids"]

            features.append(tokenized)

    print(len(features))
    return features
