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
    ElectraConfig,  # type: ignore
    ElectraForSequenceClassification,  # type: ignore
    ElectraTokenizer, # type: ignore
    get_linear_schedule_with_warmup, # type: ignore
    glue_compute_metrics, # type: ignore
    glue_output_modes, # type: ignore
    glue_processors # type: ignore
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

GLUE_TASK_TO_METRICS = {
    "cola": ["mcc"],
    "sst-2": ["acc"],
    "mrpc": ["acc", "f1"],
    "sts-b": ["pearson", "spearmanr"],
    "qqp": ["acc", "f1"],
    "mnli": ["mnli/acc"],
    "mnli-mm": ["mnli-mm/acc"],
    "qnli": ["acc"],
    "rte": ["acc"],
    "wnli": ["acc"],
}

GLUE_TASK_TO_COLUMNS = {
    "cola": ("sentence", None),
    "sst-2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "sts-b": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
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
    tokenizer_name: str = "./output/ckpt/final/tokenizer-trained.json"
    max_seq_length: int = 128
    do_train: bool = True
    do_eval: bool = True
    evaluate_during_training: bool = False
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
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
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(console=True):
    """Set up logging."""
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    split = "validation" if evaluate else "train"
    dataset_path = os.path.join(args.data_dir, split)
    
    # Handle mnli validation splits
    if task == "mnli" and evaluate:
        dataset_path = os.path.join(args.data_dir, "validation_matched")
    elif task == "mnli-mm" and evaluate:
        dataset_path = os.path.join(args.data_dir, "validation_mismatched")
        task = "mnli" # Use mnli columns for mnli-mm

    dataset = load_from_disk(dataset_path)

    col1, col2 = GLUE_TASK_TO_COLUMNS[task]
    label_col = "label"

    def preprocess_function(examples):
        if col2 is not None:
            return tokenizer(
                examples[col1], examples[col2],
                truncation=True, padding="max_length", max_length=args.max_seq_length
            )
        else:
            return tokenizer(
                examples[col1],
                truncation=True, padding="max_length", max_length=args.max_seq_length
            )

    dataset = dataset.map(preprocess_function, batched=True)


    # Force label to int64 in batched mode
    def ensure_int_label(batch):
        labels = []
        for label in batch["label"]:
            if isinstance(label, str):
                try:
                    labels.append(int(label))
                except Exception:
                    labels.append(-1)
            elif label is None or (isinstance(label, float) and np.isnan(label)):
                labels.append(-1)
            else:
                labels.append(int(label))
        batch["label"] = np.array(labels, dtype=np.int64)
        return batch
    dataset = dataset.map(ensure_int_label, batched=True)

    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in dataset.column_names:
        columns.append("token_type_ids")
    columns.append(label_col)
    
    dataset.set_format(type="torch", columns=columns)
    
    # Create a TensorDataset for compatibility with the rest of the script
    tensors = [dataset[col] for col in columns]
    return TensorDataset(*tensors) # type: ignore

def train(args, train_dataset, model, tokenizer):
    """Train the model on the training set."""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    print(t_total)
    
    # Add warmup steps if not specified
    if args.warmup_steps == 0:
        args.warmup_steps = int(t_total * 0.1) # Use 10% for warmup
        logger.info(f"Warmup steps not specified. Set to {args.warmup_steps} (10% of total steps).")

    # Prepare optimizer and schedule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args.seed)  # For reproducibility
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            
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
                    # Log metrics
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            logger.info(f"  {key} = {value}")
                    
                    logger.info(f"  Step = {global_step}, Loss = {(tr_loss - logging_loss) / args.logging_steps}")
                    logging_loss = tr_loss
                
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saving model checkpoint to {output_dir}")
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    """Evaluate the model on the evaluation set."""
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
        
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0) # type: ignore
    
    eval_loss = eval_loss / nb_eval_steps
    
    # Get task-specific processor
    processor = glue_processors[args.task_name]()
    label_list = processor.get_labels()
    
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1) # type: ignore
    elif args.output_mode == "regression":
        preds = np.squeeze(preds) # type: ignore
    
    # Compute metrics
    result = glue_compute_metrics(args.task_name, preds, out_label_ids)
    result["eval_loss"] = eval_loss
    
    # Print evaluation results
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]}")
    
    return result

def main():

    task = "cola"
    no_cuda = False

    args = Args(
        data_dir = "data/glue_" + task,
        model_name_or_path = "output/ckpt/final",
        task_name = "cola",
        output_dir = "output/electra_" + task,
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu"),
        n_gpu = torch.cuda.device_count()
    )
    
    # Make output dir
    os.makedirs(args.output_dir, exist_ok = True)
    
    # Setup logging
    setup_logging()
    
    # Set seed
    set_seed(args.seed)
    
    # Task name mapping for internal use
    args.task_name = args.task_name.lower()
    TASK_NAME_MAP = {
        "sst2": "sst-2",
        "stsb": "sts-b",
    }
    args.task_name = TASK_NAME_MAP.get(args.task_name, args.task_name)
    internal_task_name = args.task_name # for clarity, though it's now the same
    
    # Prepare GLUE task
    if internal_task_name not in glue_processors:
        raise ValueError(f"Task not found: {internal_task_name}")
    
    processor = glue_processors[internal_task_name]()
    args.output_mode = glue_output_modes[internal_task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list) if args.output_mode == "classification" else 1
    
    # Load pretrained model and tokenizer
    config = ElectraConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=internal_task_name,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_name,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        do_lower_case=True
    )
    model = ElectraForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        use_safetensors=True,
        config=config,
    )
    
    model.to(device= args.device) # type: ignore
    
    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        # Use internal_task_name for loading data
        train_dataset = load_and_cache_examples(args, internal_task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
        
        # Save the trained model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        logger.info(f"Saving model to {args.output_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        assert isinstance(model_to_save, ElectraForSequenceClassification)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # Evaluation
    results = {}
    if args.do_eval:
        # If training was not part of this run, load the specified model from disk.
        # Otherwise, the model is already in memory and has been fine-tuned.
        if not args.do_train:
            logger.info(f"Loading model for evaluation from {args.model_name_or_path}")
            model = ElectraForSequenceClassification.from_pretrained(args.model_name_or_path)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=args.tokenizer_name,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
                do_lower_case=True
            )
            model.to(device=args.device) # type: ignore

        result = evaluate(args, model, tokenizer, prefix="")
        results.update(result)
    
    return results

if __name__ == "__main__":
    main() 
