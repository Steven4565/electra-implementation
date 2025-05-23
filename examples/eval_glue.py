import os
import sys
import argparse
import logging
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report

from transformers import (
    ElectraConfig, 
    ElectraForSequenceClassification, 
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors
)

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
    """Load and preprocess GLUE task data."""
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]
    
    # Load data features
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
        if hasattr(args, 'max_eval_samples') and args.max_eval_samples is not None:
            examples = examples[:args.max_eval_samples]
    else:
        examples = processor.get_train_examples(args.data_dir)
        if hasattr(args, 'max_train_samples') and args.max_train_samples is not None:
            examples = examples[:args.max_train_samples]
    
    # Get labels
    label_list = processor.get_labels()
    
    # Convert examples to features
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        # Convert label to id
        if output_mode == "classification":
            label = label_list.index(example.label)
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise ValueError(f"Unsupported output mode: {output_mode}")
        
        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": label,
        })
    
    # Convert to tensors
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    
    if output_mode == "classification":
        all_labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f["label"] for f in features], dtype=torch.float)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train(args, train_dataset, model, tokenizer):
    """Train the model on the training set."""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
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
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    
    # Get task-specific processor
    processor = glue_processors[args.task_name]()
    label_list = processor.get_labels()
    
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    
    # Compute metrics
    result = glue_compute_metrics(args.task_name, preds, out_label_ids)
    result["eval_loss"] = eval_loss
    
    # Print evaluation results
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]}")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory containing GLUE data.")
    parser.add_argument("--model_type", type=str, default="electra",
                        help="Model type (electra)")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pre-trained model or shortcut name from huggingface.co/models")
    parser.add_argument("--task_name", type=str, required=True,
                        help="GLUE task name (e.g., cola, sst-2, mrpc, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where model checkpoints will be written.")
    
    # Other parameters
    parser.add_argument("--config_name", type=str, default="",
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default="",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="훈련 샘플 최대 개수")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="평가 샘플 최대 개수")
    
    args = parser.parse_args()
    
    # Set up output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Setup logging
    setup_logging()
    
    # Set CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    
    # Set seed
    set_seed(args.seed)
    
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in glue_processors:
        raise ValueError(f"Task not found: {args.task_name}")
    
    processor = glue_processors[args.task_name]()
    args.output_mode = glue_output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list) if args.output_mode == "classification" else 1
    
    # Load pretrained model and tokenizer
    config = ElectraConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = ElectraTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=True, # Assuming uncased model for GLUE
    )
    model = ElectraForSequenceClassification.from_pretrained(
        args.model_name_or_path, # Changed from args.output_dir
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    
    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
        
        # Save the trained model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        logger.info(f"Saving model to {args.output_dir}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # Evaluation
    results = {}
    if args.do_eval:
        evaluation_model_path = args.model_name_or_path
        # If training was also performed in the same run, the fine-tuned model would be in args.output_dir.
        # However, for a pure evaluation run (like this one), we use the model specified in model_name_or_path.
        # The original script might assume training always happens before evaluation if both flags are set.
        # For clarity, if only do_eval is true, we load from model_name_or_path.
        # If do_train was also true, it implies the model was saved to output_dir after training.
        if args.do_train: 
             evaluation_model_path = args.output_dir
             logger.info(f"Evaluating model from {evaluation_model_path} (fine-tuned in this run).")
        else:
            logger.info(f"Evaluating model from {evaluation_model_path} (pre-trained).")

        model = ElectraForSequenceClassification.from_pretrained(evaluation_model_path)
        tokenizer = ElectraTokenizer.from_pretrained(evaluation_model_path) # Load tokenizer from the same path
        model.to(args.device)
        
        result = evaluate(args, model, tokenizer, prefix="")
        results.update(result)
    
    return results

if __name__ == "__main__":
    main() 