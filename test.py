import os

import tokenizers
from tokenizer.tokenizer import train, create_corpus, load_tokenizer
from pretraining.dataset import ExampleBuilder,IterableOwtDataset

# create_corpus()
# train()



tokenizer = load_tokenizer()
print(tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt"))
# print(tokenizer.vocab["[MASK]"])

dataset = iter(IterableOwtDataset(map(lambda s: "data/" + s ,os.listdir("./data"))))
# print(next(dataset))
# print(next(dataset))


builder = ExampleBuilder(tokenizer.vocab, 256)


def get_data(): 
    while True: 
        token_ids = tokenizer(next(dataset)['text'])['input_ids']
        example = builder.add_line(token_ids)
        if (example): 
            yield example

gen2 = get_data()
# print(next(gen2))
# print(next(gen2))
# print(next(gen2))
# print(next(gen2))
