# ELECTRA Implementation 

This is a PyTorch implementation of the ELECTRA model from the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) by Clark et al.


## TODO: 
- Clean hard coded files
- Generate dataset and save to hf

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Download datasets
Run the following command to download the FULL openwebtext dataset (>50GB).
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
python pretraining/tokenizer.py
```

### Generate pretraining examples
Create 80GB worth of examples, then save to disk
```bash
python pretraining/create_examples_ds.py
```
Run the following command to generate less data for development
```bash
python pretraining/create_examples_ds.py --dev
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
  --data_dir data/glue_cola \
  --model_name_or_path output/ckpt/final \
  --task_name cola \
  --do_train \
  --do_eval \
  --output_dir output/electra_TASK \
  --tokenizer_name ./tokenizer-trained.json
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
