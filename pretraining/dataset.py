from os import makedirs
from pathlib import Path
import torch
import random
from datasets import load_from_disk, load_dataset, Dataset
from datasets.arrow_writer import ArrowWriter

from pretraining.tokenizer import load_tokenizer


class BertTrainingDataset(torch.utils.data.IterableDataset):
    def __init__(self, owt_dataset, builder): 
        self.owt_dataset = iter(owt_dataset)
        self.builder = builder
        self.tokenizer = load_tokenizer("tokenizer-trained.json")


    # Only for testing
    @staticmethod
    def tokenize(tokenizer, text): 
        tokenized = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        return ids

    def __iter__(self): 
        while True: 
            token_ids = self.tokenize(self.tokenizer, next(self.owt_dataset)["text"])
            example = self.builder.add_line(token_ids)
            if (example): 
                yield example

class ExampleBuilder:
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, vocab, max_length):
        self._vocab = vocab
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

    def add_line(self, bert_tokids):
        """Adds a line of text to the current example being built."""
        # line = line.strip().replace("\n", " ")
        # if (not line) and self._current_length != 0:  # empty lines separate docs
        #     return self._create_example()
        # bert_tokens = self._tokenizer.tokenize(line)
        # bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
        self._current_sentences.append(bert_tokids)
        self._current_length += len(bert_tokids)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                len(first_segment) + len(sentence) < first_segment_target_length or
                (len(second_segment) == 0 and
                len(first_segment) < first_segment_target_length and
                random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_tf_example(first_segment, second_segment)

    def _make_tf_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a training example"""
        vocab = self._vocab
        input_ids = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]]
        segment_ids = [0] * len(input_ids)
        if second_segment:
            input_ids += second_segment + [vocab["[SEP]"]]
            segment_ids += [1] * (len(second_segment) + 1)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids))

        def create_int_feature(tensors):
            return torch.tensor(tensors)

        tf_example = {
            "input_ids": create_int_feature(input_ids),
            "input_mask": create_int_feature(input_mask),
            "segment_ids": create_int_feature(segment_ids)
        }
        return tf_example


class HFInfiniteWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        # Buffer size is how many examples is saved in the memory to be shuffled. Default is 10k
        self.iter = iter(self.dataset.shuffle(seed=42))

    def __iter__(self): 
        while (True): 
            try:
                x = next(self.iter)
                tensor_ex = {
                    "input_ids": torch.tensor(x["input_ids"]), # type: ignore
                    "input_mask": torch.tensor(x["input_mask"]), # type: ignore
                    "segment_ids": torch.tensor(x["segment_ids"]), # type: ignore
                }
                yield tensor_ex
            except StopIteration: 
                self.iter = iter(self.dataset.shuffle())
                print("Looping dataset")

class ExampleDiskWriter:
    def __init__(self, example_dataset: BertTrainingDataset, n_per_file: int, save_dir: str, total_examples: int): 
        self.example_dataset = example_dataset
        self.n_per_file = n_per_file
        self.save_dir = save_dir
        self.file_counter = 0
        self.total_examples = total_examples
        self.total_example_counter = 0

        makedirs(save_dir, exist_ok=True)

    def write(self): 
        data_iter = iter(self.example_dataset)
        while (self.total_example_counter < self.total_examples):
            buffer = []

            while (len(buffer) < self.n_per_file):
                example = next(data_iter)
                buffer.append(example)
                self.total_example_counter += 1

            arrow_file_path = Path(self.save_dir) / Path(f"{self.file_counter}.arrow")
            self.file_counter += 1

            with ArrowWriter(path=str(arrow_file_path)) as writer: 
                buffer_keys = buffer[0].keys()
                dict_of_lists = {k: [d[k] for d in buffer] for k in buffer_keys}
                writer.write_batch(dict_of_lists)
                writer.finalize()

def example_dataset_disk_loader():
    dir = "data/preprocessed_examples/"
    files = [str(Path(dir) / f.name) for f in Path("data/preprocessed_examples/").iterdir()]
    ds = load_dataset("arrow", data_files={"train": files}, split="train")
    assert isinstance(ds, Dataset)
    inf = HFInfiniteWrapper(ds)
    return inf

