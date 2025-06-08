import os
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from pretraining.dataset import BertTrainingDataset, ExampleBuilder, ExampleDiskWriter, HFInfiniteWrapper, example_dataset_disk_loader
from pretraining.tokenizer import load_tokenizer, PreTrainedTokenizerFast
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.electra import ElectraForSequenceClassification, ElectraForPreTraining, ElectraPreTrainedModel


# tokenizer = load_tokenizer("tokenizer-trained.json")
# vocab = tokenizer.vocab
#
# owt_dataset = load_from_disk("./data/openwebtext/") 
# if (isinstance(owt_dataset, DatasetDict)): 
#     owt_dataset = owt_dataset['train']
# dataset = HFInfiniteWrapper(owt_dataset)
# builder = ExampleBuilder(vocab, 128)
#
# bert_dataset = BertTrainingDataset(dataset, builder)
# iter_d = iter(bert_dataset)

def test_bert_dataset(): 
    # for i in range(10): 
    #     text = next(iter_d)
    #     print('================')
    #     print(tokenizer.convert_ids_to_tokens(text["input_ids"]))

    dataset = load_from_disk("./data/openwebtext/").select([0 ,1, 2]).repeat(10000)
    iter_d = iter(dataset)

    print(next(iter_d))
    print(next(iter_d))
    print(next(iter_d))
    print(next(iter_d))

def test_preprocessing():
    writer = ExampleDiskWriter(bert_dataset, 1000, "data/preprocessed_examples/", 5000)
    writer.write()

def test_read_preprocessed(): 
    ds = example_dataset_disk_loader()
    iter_d = iter(ds)
    print(next(iter_d))
    print(next(iter_d))
    print(next(iter_d))

# test_preprocessing()
# test_read_preprocessed()

