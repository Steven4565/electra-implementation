import argparse
from pretraining.dataset import ExampleDiskWriter
from pretraining.dataset import BertTrainingDataset, ExampleBuilder, ExampleDiskWriter, HFInfiniteWrapper
from pretraining.tokenizer import load_tokenizer

def main(dev = False): 
    tokenizer = load_tokenizer("tokenizer-trained.json")
    vocab = tokenizer.vocab

    dataset = HFInfiniteWrapper("./data/openwebtext/")
    builder = ExampleBuilder(vocab, 128)

    bert_dataset = BertTrainingDataset(dataset, builder)
    print(dev)

    if (dev == False):
        # 25M examples for 80GB data (each file is 3.21kB)
        # For 500MB per file, use 150k data 
        writer = ExampleDiskWriter(bert_dataset, 150000, "data/preprocessed_examples/", 25000000)
        writer.write()
    else: 
        writer = ExampleDiskWriter(bert_dataset, 100, "data/preprocessed_examples/", 10000)
        writer.write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dev", action="store_true", help="Download a subset of the dataset for development")
    args = parser.parse_args()

    main(args.dev)
