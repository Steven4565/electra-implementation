from pretraining.dataset import BertTrainingDataset, ExampleBuilder, HFInfiniteWrapper
from pretraining.tokenizer import load_tokenizer


tokenizer = load_tokenizer("tokenizer-trained.json")
vocab = tokenizer.vocab

dataset = HFInfiniteWrapper("./data/openwebtext/")
builder = ExampleBuilder(vocab, 128)

bert_dataset = BertTrainingDataset(dataset, builder)
iter_d = iter(bert_dataset)
for i in range(10): 
    text = next(iter_d)
    print('================')
    # print(tokenizer.convert_ids_to_tokens(text["input_ids"]))

# dataset = load_from_disk("./data/openwebtext/").select([0 ,1, 2]).repeat(10000)
# iter_d = iter(dataset)

# print(next(iter_d))
# print(next(iter_d))
# print(next(iter_d))
# print(next(iter_d))

