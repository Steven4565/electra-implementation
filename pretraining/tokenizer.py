import os
from torch import std
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

def create_corpus(dataset_dir, n_data, corpus_file):
    dataset = iter(load_from_disk(dataset_dir))

    for i in range(5): 
        print(next(dataset))

    with open(corpus_file, "w", encoding="utf-8") as f:
        for _ in range(n_data):
            try:
                line = next(dataset)["text"]
                if line.strip():
                    f.write(line.strip() + "\n")
            except StopIteration:
                break

def train(corpus_file_dir, saved_tokenizer_dir):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]")) # type: ignore
    tokenizer.pre_tokenizer = BertPreTokenizer() # type: ignore

    trainer = WordPieceTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        vocab_size=30000,
        min_frequency=2,
        limit_alphabet=1000
    )

    tokenizer.train(files=[corpus_file_dir], trainer=trainer)
    tokenizer.save(saved_tokenizer_dir)

def load_tokenizer(saved_tokenizer_dir): 
    return PreTrainedTokenizerFast(tokenizer_file=saved_tokenizer_dir)

def main(): 
    saved_tokenizer_dir = "./tokenizer-trained.json"
    dataset_dir = "./data/openwebtext/"
    corpus_file = "./data/corpus.txt"
    n_data = 1_000_000
    create_corpus(dataset_dir, n_data, corpus_file)
    train(corpus_file, saved_tokenizer_dir)

if __name__ == "__main__":
    main() 
