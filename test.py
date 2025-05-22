import os
import shutil
import numpy as np
import torch
import tokenizers
from tokenizer.tokenizer import train, create_corpus, load_tokenizer
from pretraining.dataset import BertTrainingDataset, ExampleBuilder,IterableOwtDataset, cycle_generator
from pretraining.preprocess import run_processes


def test_train_corpus(): 
    create_corpus()
    train()



def test_get_examples(): 
    tokenizer = load_tokenizer()
    print(tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt"))
    print(tokenizer.vocab["[MASK]"])

    dataset = iter(IterableOwtDataset(map(lambda s: "data/" + s ,os.listdir("./data"))))
    print(next(dataset))
    print(next(dataset))


    builder = ExampleBuilder(tokenizer.vocab, 256)


    def get_data(): 
        while True: 
            token_ids = tokenizer(next(dataset)['text'])['input_ids']
            example = builder.add_line(token_ids)
            if (example): 
                yield example

    gen2 = get_data()
    print(next(gen2))
    print(next(gen2))
    print(next(gen2))
    print(next(gen2))

def test_preprocessing(): 
    try: 
        shutil.rmtree("./data_preprocessed/")
    except:
        pass
    tokenizer = load_tokenizer()
    run_processes(tokenizer, 2, 100)

def test_preprocessing_results():
    tokenizer = load_tokenizer()
    src_dir = "./data_preprocessed/"
    file = os.listdir(src_dir)[0]
    data0 = torch.load(src_dir + file, weights_only=False)[0]
    print(tokenizer.decode(data0))


def test_training_dataset(src_dir = "./data_preprocessed/"): 
    tokenizer = load_tokenizer()
    owt_dataset_gen = cycle_generator(iter(IterableOwtDataset(map(lambda s: src_dir + s, os.listdir(src_dir)))))
    builder = ExampleBuilder(tokenizer.vocab, 256)

    train_dataset = iter(BertTrainingDataset(owt_dataset_gen, builder))
    print(next(train_dataset))

test_training_dataset()
