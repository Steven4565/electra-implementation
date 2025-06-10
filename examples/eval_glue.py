import os
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm, trange
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Optional

from transformers import (
    ElectraConfig,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import evaluate
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# same as before
GLUE_TASK_TO_COLUMNS = {
    "cola": ("sentence", None),
    "sst-2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "sts-b": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

@dataclass
class Args:
    data_dir: str
    model_name_or_path: str
    task_name: str
    output_dir: str
    device: torch.device
    n_gpu: int
    output_mode: str = ""
    config_name: str = ""
    tokenizer_name: str = "./output/electra_pretrain/ckpt/final/tokenizer-trained.json"
    max_seq_length: int = 128
    do_train: bool = True
    do_eval: bool = True
    evaluate_during_training: bool = False
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 5.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 500
    no_cuda: bool = False
    seed: int = 42
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(console=True):
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if console:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        logger.addHandler(h)

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    split = "validation" if evaluate else "train"
    path = os.path.join(args.data_dir, split)
    if task == "mnli" and evaluate:
        path = os.path.join(args.data_dir, "validation_matched")
    elif task == "mnli-mm" and evaluate:
        path = os.path.join(args.data_dir, "validation_mismatched")
        task = "mnli"

    dataset = load_from_disk(path)
    col1, col2 = GLUE_TASK_TO_COLUMNS[task]
    maxlen = args.max_seq_length

    def preprocess(examples):
        if col2:
            return tokenizer(examples[col1], examples[col2],
                             truncation=True, padding="max_length", max_length=maxlen)
        return tokenizer(examples[col1],
                         truncation=True, padding="max_length", max_length=maxlen)

    dataset = dataset.map(preprocess, batched=True)

    def ensure_int_label(batch):
        labs = []
        for l in batch["label"]:
            if isinstance(l, str):
                try: labs.append(int(l))
                except: labs.append(-1)
            elif l is None or (isinstance(l, float) and np.isnan(l)):
                labs.append(-1)
            else:
                labs.append(int(l))
        batch["label"] = np.array(labs, dtype=np.int64)
        return batch

    dataset = dataset.map(ensure_int_label, batched=True)

    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in dataset.column_names:
        cols.append("token_type_ids")
    cols.append("label")

    dataset.set_format(type="torch", columns=cols)
    return TensorDataset(*(dataset[c] for c in cols))

def train(args, train_dataset, model, tokenizer):
    sampler = RandomSampler(train_dataset)
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(loader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_steps == 0:
        args.warmup_steps = int(t_total * 0.1)
        logger.info(f"Warmup steps set to {args.warmup_steps}")

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args.seed)

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer) # type: ignore
                        for k,v in results.items():
                            logger.info(f"  {k} = {v}")
                    logger.info(f"  Step = {global_step}, Loss = {(tr_loss - logging_loss)/args.logging_steps}")
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    out_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(out_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    logger.info(f"Saved checkpoint to {out_dir}")

                if args.max_steps > 0 and global_step > args.max_steps:
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step

def evaluate_model(args, model, tokenizer, prefix=""):
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    sampler = SequentialSampler(eval_dataset)
    loader = DataLoader(eval_dataset, sampler=sampler, batch_size=args.eval_batch_size)

    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    eval_loss = 0.0
    nb_steps = 0
    preds = None
    labels = None

    for batch in tqdm(loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch[2]

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_loss, logits = (outputs.loss, outputs.logits) if hasattr(outputs, "loss") else (outputs[0], outputs[1])
            eval_loss += tmp_loss.mean().item()

        nb_steps += 1
        arr = logits.detach().cpu().numpy()
        lab = inputs["labels"].detach().cpu().numpy()
        preds = arr if preds is None else np.append(preds, arr, axis=0)
        labels = lab if labels is None else np.append(labels, lab, axis=0)

    eval_loss /= nb_steps


    # post-process predictions
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1) # type: ignore
    else: 
        preds = np.squeeze(preds) # type: ignore


    # compute metrics via 🤗 Evaluate
    metric = evaluate.load("glue", args.task_name)
    result = metric.compute(predictions=preds, references=labels)
    result["eval_loss"] = eval_loss

    logger.info(f"***** Eval results {prefix} *****")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]}")

    return result

def main():
    task = "cola"
    no_cuda = False

    args = Args(
        data_dir    = "data/glue_" + task,
        model_name_or_path = "output/electra_pretrain/ckpt/final/",
        task_name   = task,
        output_dir  = "output/electra_" + task,
        device      = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu"),
        n_gpu       = torch.cuda.device_count(),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging()
    set_seed(args.seed)

    # normalize task name
    tmap = {"sst2":"sst-2", "stsb":"sts-b"}
    args.task_name = tmap.get(args.task_name.lower(), args.task_name.lower())

    # set output mode
    args.output_mode = "regression" if args.task_name == "sts-b" else "classification"

    # infer num_labels from the train split
    tmp = load_from_disk(os.path.join(args.data_dir, "train"))
    feat = tmp.features["label"]
    if args.output_mode == "classification" and hasattr(feat, "num_classes"):
        num_labels = feat.num_classes
    else:
        num_labels = 1

    # load model + tokenizer
    config = ElectraConfig.from_pretrained(
        args.config_name or args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_name,
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", do_lower_case=True,
    )
    model = ElectraForSequenceClassification.from_pretrained(
        args.model_name_or_path, use_safetensors=True, config=config
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_ds = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_ds, model, tokenizer)
        logger.info(f" global_step = {global_step}, avg loss = {tr_loss}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    results = {}
    if args.do_eval:
        if not args.do_train:
            # reload fresh
            model = ElectraForSequenceClassification.from_pretrained(args.model_name_or_path)
            model.to(args.device)
        results = evaluate_model(args, model, tokenizer)

    return results

if __name__ == "__main__":
    main()
