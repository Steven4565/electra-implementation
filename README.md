# ELECTRA Implementation 

This is a PyTorch implementation of the ELECTRA model from the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) by Clark et al.

## Preprocessing pretraining dataset

### Download OWT dataset
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

## Pretraining 

### Download pretraining dataset
Create the pretraining dataset from the instructions above or download from the following HuggingFace repo along with the tokenizer: 
```bash
huggingface-cli download JEEHANA/AML-Electra --repo-type dataset --pattern "data/*"
huggingface-cli download JEEHANA/AML-Electra --repo-type dataset --pattern "trained-tokenizer.json"
```
Make sure to put the datasets inside of `data/` and the `trained-tokenizer.json` at the root dir.

### Run the pretraining code
```bash
python pretraining/pretrain.py
```


## Finetuning

### Download our pretrained model, dataset, and tokenizer
Run the following bash file to download the eval dataset, our pretrained model, and tokenizer.
```bash
bash finetune-setup.sh
```
Alternatively, you can also downlaod the full OWT and GLUE dataset with the following command (add the `--glue` flag to download the full Glue benchmark dataset)
```bash
python pretraining/download_datasets.py --glue
```

### Evaluation on GLUE Tasks
Configure config inside `main` function in `eval_glue.py` first
```bash
python examples/eval_glue.py
```

## Output Results

After pretraining is complete, you can find the model checkpoints in the output directory specified with the `--output_dir` parameter:

- `output/electra_pretrain/ckpt/final`: Final model checkpoint
- `output/electra_pretrain/ckpt/{step}`: Intermediate model checkpoints at specified steps

After fine-tuning, you can find the results in a folder at `output/electra_{GLUE_TASK}/`.
Example for CoLA:
- `output/electra_cola/`

## References

- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB)
- [Original ELECTRA code (TensorFlow)](https://github.com/google-research/electra)
- [Reference PyTorch implementation](https://github.com/lucidrains/electra-pytorch) 
