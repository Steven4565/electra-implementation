import argparse

from datasets import DatasetDict
from pretraining.dataset import ExampleDiskWriter, load_from_disk
from pretraining.dataset import BertTrainingDataset, ExampleBuilder, ExampleDiskWriter, HFInfiniteWrapper
from pretraining.tokenizer import load_tokenizer

def main(dev = False): 
    tokenizer = load_tokenizer("tokenizer-trained.json")
    vocab = tokenizer.vocab

    
    owt_dataset = load_from_disk("./data/openwebtext/") 
    if (isinstance(owt_dataset, DatasetDict)): 
        owt_dataset = owt_dataset['train']
    builder = ExampleBuilder(vocab, 128)

    bert_dataset = BertTrainingDataset(owt_dataset, builder)
    
    if (dev): 
        print("Generating development example dataset")
    else:
        print("Generating production example dataset")

    if (dev == False):
        # 25M examples for 80GB data (each file is 3.21kB)
        # For 500MB per file, use 150k data 
        writer = ExampleDiskWriter(bert_dataset, 150000, "data/preprocessed_examples/", 25000000)
        writer.write()
    else: 
        writer = ExampleDiskWriter(bert_dataset, 1000, "data/preprocessed_examples/", 10000)
        writer.write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dev", action="store_true", help="Download a subset of the dataset for development")
    args = parser.parse_args()

    main(args.dev)
