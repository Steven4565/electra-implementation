import os
import multiprocessing
from itertools import batched


def run_processes(tokenizer, processes):
    src_dir = ""
    dest_dir = ""

    files = os.listdir(src_dir)
    batched_files = list(batched(files, processes))

    pool = multiprocessing.Pool(processes)
    for batch in batched_files:
        job = tokenize_process(batch)


    

def tokenize_process(files):
    pass


