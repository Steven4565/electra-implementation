import os
import sys
import argparse
import logging
import glob
from pathlib import Path
import json
import random
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pretraining.tokenization import FullTokenizer

logger = logging.getLogger(__name__)

def setup_logging(console=True):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def download_and_save_huggingface_dataset(dataset_name, output_dir, split="train", max_lines=5000):
    """Downloads a dataset from Hugging Face, takes a subset, and saves it to text files."""
    logger.info(f"Downloading {dataset_name} from Hugging Face datasets...")
    # For Wikipedia, we might need to specify a configuration like '20220301.en'
    # Adjust dataset loading based on its structure.
    # For simplicity, we'll try to load the plain text version if available.
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True) # Use streaming to avoid downloading everything
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        logger.info("Attempting to load with a specific configuration for Wikipedia (e.g., '20220301.en')")
        try:
            dataset = load_dataset(dataset_name, "20220301.en", split=split, streaming=True)
        except Exception as e_config:
            logger.error(f"Error loading dataset {dataset_name} with configuration: {e_config}")
            return None

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "huggingface_data.txt")
    
    lines_written = 0
    logger.info(f"Writing a maximum of {max_lines} lines to {output_file_path}...")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Processing dataset"):
            if lines_written >= max_lines:
                break
            # Adapt this part based on the actual structure of the dataset
            # Assuming the dataset has a 'text' field.
            text = example.get("text", "") 
            if isinstance(text, str) and text.strip():
                # Write each sentence or paragraph on a new line if text contains multiple.
                # For this example, we assume 'text' is a single block.
                for line in text.split('\\n'): # Split if text has internal newlines
                    if line.strip():
                        f.write(line.strip() + '\\n')
                        lines_written +=1
                        if lines_written >= max_lines:
                            break
            if lines_written >= max_lines:
                break
                
    logger.info(f"Finished writing {lines_written} lines to {output_file_path}")
    return output_file_path


def convert_to_text_files(input_files, output_dir, lines_per_file=100000):
    """Convert raw text data to formatted text files for training."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_lines = 0
    file_num = 0
    current_file_lines = 0
    out_file = None
    
    input_files_list = glob.glob(input_files)
    random.shuffle(input_files_list)
    
    logger.info(f"Processing {len(input_files_list)} input files...")
    
    # Get a count of total lines for tqdm progress bar
    total_lines_count = 0
    for fname in tqdm(input_files_list, desc="Counting lines"):
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                total_lines_count += 1
    
    logger.info(f"Total lines to process: {total_lines_count}")
    
    # Process files
    with tqdm(total=total_lines_count, desc="Processing lines") as pbar:
        for fname in input_files_list:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if current_file_lines == 0:
                        if out_file is not None:
                            out_file.close()
                        out_path = os.path.join(output_dir, f"text_{file_num}.txt")
                        out_file = open(out_path, 'w', encoding='utf-8')
                        file_num += 1
                    
                    line = line.strip()
                    if line:  # Skip empty lines
                        out_file.write(line + '\n')
                        current_file_lines += 1
                        total_lines += 1
                    
                    if current_file_lines >= lines_per_file:
                        current_file_lines = 0
                    
                    pbar.update(1)
    
    if out_file is not None:
        out_file.close()
    
    logger.info(f"Processed {total_lines} lines into {file_num} files in {output_dir}")


def create_vocab_file(input_dir, output_file, vocab_size=30522):
    """Create a vocabulary file from the processed text data.
    
    This is a simplified version that just uses a default BERT vocab.
    For a real application, you would build the vocab from the data.
    """
    
    logger.info("Creating vocabulary file...")
    
    # Default BERT special tokens
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ]
    
    # Write the vocab file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write special tokens
        for token in special_tokens:
            f.write(token + '\n')
        
        # For simplicity, fill the rest of the vocab with dummy tokens
        # In a real application, you would compute the most frequent tokens
        for i in range(vocab_size - len(special_tokens)):
            f.write(f"[unused{i}]\n")
    
    logger.info(f"Created vocabulary file with {vocab_size} tokens at {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess text data for ELECTRA pretraining")
    parser.add_argument("--input_files", type=str, required=False, default=None,
                        help="Path pattern to raw input files (e.g., 'data/raw/*.txt'). If not provided, downloads from Hugging Face.")
    parser.add_argument("--huggingface_dataset", type=str, default="wikipedia",
                        help="Name of the dataset to download from Hugging Face (e.g., 'wikipedia', 'bookcorpus')")
    parser.add_argument("--huggingface_dataset_config", type=str, default="20220301.en",
                        help="Configuration for the Hugging Face dataset if needed (e.g., '20220301.en' for wikipedia)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed text files and downloaded data")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Output path for vocabulary file")
    parser.add_argument("--max_lines", type=int, default=5000,
                        help="Maximum number of lines to process from the dataset (default: 5000)")
    parser.add_argument("--lines_per_file", type=int, default=100000,
                        help="Number of lines per output file for ELECTRA pretraining format")
    parser.add_argument("--vocab_size", type=int, default=30522,
                        help="Size of vocabulary (default: 30522 for BERT compatibility)")
    args = parser.parse_args()
    
    setup_logging()
    
    # Create the output directory if it doesn't exist
    # The output_dir will be used for downloaded data and then for processed text files.
    # Let's make a subdirectory for the initial download to keep things clean.
    raw_data_download_dir = os.path.join(args.output_dir, "raw_downloaded_data")
    os.makedirs(raw_data_download_dir, exist_ok=True)

    processed_text_output_dir = os.path.join(args.output_dir, "processed_for_pretraining")
    os.makedirs(processed_text_output_dir, exist_ok=True)

    input_files_to_process = args.input_files

    if not input_files_to_process:
        logger.info(f"No input_files provided. Attempting to download from Hugging Face: {args.huggingface_dataset}")
        downloaded_file_path = download_and_save_huggingface_dataset(
            dataset_name=args.huggingface_dataset,
            output_dir=raw_data_download_dir,
            split="train",
            max_lines=args.max_lines
        )
        if downloaded_file_path:
            input_files_to_process = os.path.join(raw_data_download_dir, "*.txt")
            logger.info(f"Using downloaded data from: {input_files_to_process}")
        else:
            logger.error("Failed to download or find data from Hugging Face. Exiting.")
            return
    else:
        logger.info(f"Using provided input_files pattern: {input_files_to_process}")

    # Convert raw files to text format
    convert_to_text_files(
        input_files_to_process, 
        processed_text_output_dir, # Output to a specific subdirectory
        lines_per_file=args.lines_per_file 
        # max_files argument removed from this function call
    )
    
    # Create vocabulary file using the processed text data
    create_vocab_file(processed_text_output_dir, args.vocab_file, args.vocab_size)


if __name__ == "__main__":
    main() 