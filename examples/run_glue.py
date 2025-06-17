import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm, trange
from datasets import load_dataset
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def train(args, model, train_dataset, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args.seed)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            outputs = model(**batch)
            loss = outputs.loss

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
                    logger.info(f"Step {global_step}: Loss = {loss.item()}")


                if args.save_steps > 0 and global_step % args.save_steps == 0:
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

def run_evaluation(args, model, eval_dataset, metric, is_regression):
    """ Evaluate the model """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs.loss, outputs.logits

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if is_regression:
        predictions = np.squeeze(preds)
    else:
        predictions = np.argmax(preds, axis=1)

    result = metric.compute(predictions=predictions, references=out_label_ids)
    result["eval_loss"] = eval_loss

    return result

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", type=str, required=True, help="The name of the GLUE task to train on.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path to pre-trained tokenizer or shortcut name if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)

    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Task setup
    args.task_name = args.task_name.lower()
    if args.task_name not in GLUE_TASK_TO_KEYS:
        raise ValueError(f"Task not found: {args.task_name}")

    # Load datasets
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    # Load pretrained model and tokenizer
    tokenizer_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    
    if os.path.isdir(tokenizer_path):
        # Default to standard AutoTokenizer loading if it's a directory
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        # If a specific file is provided, use PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    model.to(args.device)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[args.task_name]
    def preprocess_function(examples):
        tokenizer_args = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*tokenizer_args, padding="max_length", max_length=args.max_seq_length, truncation=True)
        return result
    
    column_names = raw_datasets["train"].column_names
    column_names.remove("label")
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=column_names)
    processed_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"] + (["token_type_ids"] if "token_type_ids" in processed_datasets["train"].features else []))

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    metric = evaluate.load("glue", args.task_name)

    if args.do_train:
        train(args, model, train_dataset, tokenizer)
        # Save the final model
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Final model saved to {args.output_dir}")

    if args.do_eval:
        logger.info("*** Evaluate ***")
        # Load the best checkpoint for evaluation
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        model.to(args.device)
        eval_result = run_evaluation(args, model, eval_dataset, metric, is_regression)
        
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

if __name__ == "__main__":
    main() 