# MyELECTRA: ELECTRA Implementation in PyTorch

This is a PyTorch implementation of the ELECTRA model from the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) by Clark et al.

## Overview

ELECTRA is a new method for self-supervised language representation learning. Instead of masking tokens like in BERT, ELECTRA trains two transformer models: a generator and a discriminator. The generator replaces tokens with plausible alternatives sampled from its output distribution, and the discriminator predicts whether each token was replaced by the generator or not.

This approach, called replaced token detection, is more sample efficient than masked language modeling (MLM) used in BERT. The task is defined over all input tokens rather than just a small subset (e.g., 15% for BERT), so the model learns from more training signals per example.


## TODO: 
- generate vocab file
- create dataset downloader for testing vs full training

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Download datasets
Run the following command to download the FULL openwebtext dataset.
```bash
python pretraining/download_datasets.py
```
For development, download the subset of openwebtext with 
```bash
python pretraining/download_datasets.py --dev
```


### Train tokenizer
Run the following command to train a tokenizer
```bash
python pretraining/tokenizer.py \
  --dataset_dir data/openwebtext/ \
  --vocab_file data/vocab.txt
```

### Pretraining

```bash
python pretraining/pretrain.py \
  --data_dir data/text_data \
  --data_vocab_file data/vocab.txt \
  --output_dir output/electra_pretrain
```

### Evaluation on GLUE Tasks

```bash
python examples/eval_glue.py \
  --data_dir path/to/glue_data/TASK \
  --model_name_or_path output/electra_pretrain/ckpt/final \
  --task_name TASK \
  --do_train \
  --do_eval \
  --output_dir output/electra_TASK
```

Where `TASK` is one of the GLUE tasks (e.g., `cola`, `sst-2`, `mrpc`, etc.).

## Pretrained Models

After pretraining is complete, you can find the model checkpoints in the output directory specified with the `--output_dir` parameter:

- `output/electra_pretrain/ckpt/final`: Final model checkpoint
- `output/electra_pretrain/ckpt/{step}`: Intermediate model checkpoints at specified steps

## References

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)
- [Original ELECTRA code (TensorFlow)](https://github.com/google-research/electra)
- [Reference PyTorch implementation](https://github.com/lucidrains/electra-pytorch) 
