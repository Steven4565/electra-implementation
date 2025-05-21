import os
from torch import std
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from pretraining.dataset import IterableOwtDataset

corpus_file = "corpus.txt"
data_dir = "./data/"
saved_tokenizer_dir = "./tokenizer-trained.json"

def create_corpus():
    dataset_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".pt")]
    dataset_iter = iter(IterableOwtDataset(dataset_files))

    with open(corpus_file, "w", encoding="utf-8") as f:
        for _ in range(900):
            try:
                line = next(dataset_iter)["text"]
                if line.strip():
                    f.write(line.strip() + "\n")
            except StopIteration:
                break

def train():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]")) # type: ignore
    tokenizer.pre_tokenizer = BertPreTokenizer() # type: ignore

    trainer = WordPieceTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        vocab_size=30000,
        min_frequency=2,
        limit_alphabet=1000
    )

    tokenizer.train(files=[corpus_file], trainer=trainer)
    tokenizer.save(saved_tokenizer_dir)

def load_tokenizer(): 
    return PreTrainedTokenizerFast(tokenizer_file=saved_tokenizer_dir)

