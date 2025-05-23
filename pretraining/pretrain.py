import os
import sys
import logging
import random
import time
from dataclasses import dataclass
import shutil

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, ElectraForMaskedLM, ElectraForPreTraining

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myelectra_pytorch import Electra
from pretraining.arg import Str, Int, Float, Bool, parse_args
from pretraining.dataset import load_text_dataset, wrap_example_builder, new_tokenizer

logger = logging.getLogger(__name__)

########################################################################################################
## args

@dataclass
class Args:
    # Data settings
    data_dir: Str = 'data/text_data'
    data_vocab_file: Str = 'data/vocab.txt'
    data_max_seq_length: Int = 128
    
    # Output settings
    output_dir: Str = 'output'
    
    # GPU settings
    gpu: Int = 0
    gpu_enabled: Bool = True
    gpu_deterministic: Bool = False
    gpu_mixed_precision: Bool = False
    distributed_enabled: Bool = False
    distributed_world_size: Int = 1
    distributed_port: Int = 8888

    # Model settings
    model_generator: Str = 'pretraining/small_generator.json'
    model_discriminator: Str = 'pretraining/small_discriminator.json'
    model_mask_prob: Float = 0.15
    
    # Optimization settings
    opt_lr: Float = 5e-4
    opt_batch_size: Int = 32
    opt_warmup_steps: Int = 10_000
    opt_num_training_steps: Int = 100_000
    
    # Logging and checkpoint settings
    step_log: Int = 10
    step_ckpt: Int = 5_000


########################################################################################################
## train

def train(rank, args):
    #######################
    ## distributed

    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    
    if args.gpu_enabled and torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        set_gpus(rank)
    else:
        device = torch.device('cpu')
        if args.gpu_enabled and not torch.cuda.is_available():
            logger.warning("GPU is enabled but not available. Falling back to CPU.")
        elif not args.gpu_enabled:
            logger.info("GPU is disabled. Using CPU.")

    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0


    #######################
    ## preamble

    set_seed(rank)
    set_cuda(deterministic=args.gpu_deterministic, use_cuda=args.gpu_enabled and torch.cuda.is_available())

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/ckpt', exist_ok=True)

    setup_logging(filename=f'{output_dir}/output.log', console=is_master)


    #######################
    ## dataset

    tokenizer = new_tokenizer(vocab_file=args.data_vocab_file)
    vocab_size = len(tokenizer.vocab)
    
    logger.info(f"Loading dataset from {args.data_dir}")
    ds_train = load_text_dataset(data_dir=args.data_dir, tokenizer=tokenizer)
    ds_train = wrap_example_builder(ds_train, vocab=tokenizer.vocab, max_length=args.data_max_seq_length)

    pad_token_id = tokenizer.vocab['[PAD]']
    mask_token_id = tokenizer.vocab['[MASK]']
    cls_token_id = tokenizer.vocab['[CLS]']
    sep_token_id = tokenizer.vocab['[SEP]']

    assert pad_token_id == 0
    assert cls_token_id == 2
    assert sep_token_id == 3
    assert mask_token_id == 4

    def collate_batch(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        input_mask = torch.nn.utils.rnn.pad_sequence([example['input_mask'] for example in examples], batch_first=True, padding_value=pad_token_id)
        segment_ids = torch.nn.utils.rnn.pad_sequence([example['segment_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        return input_ids, input_mask, segment_ids

    ds_train_loader = DataLoader(ds_train, batch_size=args.opt_batch_size, collate_fn=collate_batch)


    #######################
    ## model

    def to_distributed_model(model):
        return model if not args.distributed_enabled else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    def tie_weights(generator, discriminator):
        generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings

    class LogitsAdapter(torch.nn.Module):
        def __init__(self, adaptee):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]
    
    logger.info("Building generator and discriminator models")
    generator = ElectraForMaskedLM(AutoConfig.from_pretrained(args.model_generator))
    discriminator = ElectraForPreTraining(AutoConfig.from_pretrained(args.model_discriminator))

    tie_weights(generator, discriminator)

    model = to_distributed_model(Electra(
        LogitsAdapter(generator),
        LogitsAdapter(discriminator),
        num_tokens=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        mask_prob=args.model_mask_prob,
        mask_ignore_token_ids=[tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
        random_token_prob=0.0).to(device))


    #######################
    ## optimizer

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)


    #######################
    ## train

    logger.info("Starting training")
    t = time.time()
    steps_s = 0.
    eta_m = 0
    data_iter = iter(ds_train_loader)

    for step in range(args.opt_num_training_steps + 1):
        try:
            input_ids, input_mask, segment_ids = next(data_iter)
        except StopIteration:
            # Restart the data iterator when it's exhausted
            data_iter = iter(ds_train_loader)
            input_ids, input_mask, segment_ids = next(data_iter)

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        assert input_ids.shape[1] <= args.data_max_seq_length

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

        if step > 0 and step % 100 == 0:
            t2 = time.time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        if step > 0 and step % args.step_ckpt == 0 and is_master:
            discriminator.electra.save_pretrained(f'{args.output_dir}/ckpt/{step}')
            logger.info(f"Saved checkpoint at step {step}")

    # Save final model
    if is_master:
        final_save_path = f'{args.output_dir}/ckpt/final'
        os.makedirs(final_save_path, exist_ok=True)
        discriminator.electra.save_pretrained(final_save_path)
        # tokenizer.save_pretrained(final_save_path)
        # Manually copy the vocab file to the final save path
        shutil.copyfile(args.data_vocab_file, os.path.join(final_save_path, "vocab.txt"))
        logger.info(f"Training complete. Final model saved to {final_save_path}. Vocab file copied.")

########################################################################################################
## utilities

def set_gpus(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    else:
        logger.info("CUDA not available or not enabled, skipping cuDNN settings.")


def setup_logging(filename, console=True):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def main():
    # Parse arguments
    args = parse_args(Args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up distributed training
    if args.distributed_enabled:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        torch.multiprocessing.spawn(train, nprocs=args.distributed_world_size, args=(args,))
    else:
        train(args.gpu, args)


if __name__ == "__main__":
    main() 