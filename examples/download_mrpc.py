import os
from datasets import load_dataset
import pandas as pd

def download_and_save_mrpc(output_dir="./data/glue_data/mrpc"):
    """Downloads MRPC dataset from Hugging Face datasets and saves as tsv files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        # Load MRPC dataset
        print("Downloading MRPC dataset from Hugging Face...")
        dataset = load_dataset('glue', 'mrpc')
        print("Download complete.")

        # Save splits as tsv files
        for split_name, split_data in dataset.items():
            df = pd.DataFrame(split_data)
            # GLUE tsv format typically has specific column names and order.
            # For MRPC, the common columns are 'Quality', '#1 String', '#2 String' for train/dev
            # and 'index', '#1 String', '#2 String' for test.
            # We need to ensure our DataFrame matches this or adapt eval_glue.py if it expects specific headers.
            
            # Let's try to save with a generic header that _read_tsv in transformers might ignore or handle.
            # Or, more robustly, ensure standard GLUE format.
            # For MRPC, from transformers.data.processors.glue.MrpcProcessor._create_examples:
            # For train/dev: sentence1, sentence2, label
            # For test: index, sentence1, sentence2
            
            output_path = os.path.join(output_dir, f"{split_name}.tsv")
            
            if split_name == 'test':
                # Test set requires 'index' and the two sentences. It doesn't have labels.
                df_to_save = pd.DataFrame({
                    'index': range(len(df)), # Add index column
                    '#1 String': df['sentence1'],
                    '#2 String': df['sentence2']
                })
            else: # train and validation
                df_to_save = pd.DataFrame({
                    'Quality': df['label'], # Label column
                    '#1 String': df['sentence1'],
                    '#2 String': df['sentence2']
                })
                # The _read_tsv method in transformers often expects a header row,
                # but the MrpcProcessor specifically skips the header.
                # Let's save without header and index for train/dev to be safe,
                # or ensure the processor handles it.
                # Based on `_create_examples` in `MrpcProcessor`, it expects lines like:
                # label\tsentence1\tsentence2 (for train/dev)
            
            print(f"Saving {split_name} data to {output_path}...")
            
            # The _read_tsv in processors.utils typically uses csv.reader with delimiter='\t' and quotechar='\"'
            # It also skips the header line (lines[1:]) if not for qnli or mnli.
            # For MRPC, the processor expects specific columns after splitting by tab.
            # Let's ensure the columns are in the order expected by MrpcProcessor:
            # For train/dev: label, sentence1, sentence2
            # For test: index, sentence1, sentence2
            
            if split_name == 'test':
                # Test set requires 'index' and the two sentences.
                # The processor expects test file format as: index\tsentence1\tsentence2
                 df_export = pd.DataFrame({
                    'index': range(len(df)),
                    'sentence1': df['sentence1'],
                    'sentence2': df['sentence2']
                })
                 df_export.to_csv(output_path, sep='\t', index=False, header=True) # Transformers test readers usually expect a header
            else: # train and validation
                # Processor expects lines with format: Quality\t#1 ID\t#2 ID\t#1 String\t#2 String
                # We need to create dummy columns for #1 ID and #2 ID as they are skipped but expected in terms of index
                df_export = pd.DataFrame({
                    'Quality': df['label'],
                    '#1 ID': ['dummy_id1'] * len(df), # Dummy column
                    '#2 ID': ['dummy_id2'] * len(df), # Dummy column
                    '#1 String': df['sentence1'],
                    '#2 String': df['sentence2']
                })
                # The processor skips the header for MRPC train/dev.
                # Order of columns for writing: Quality, #1 ID, #2 ID, #1 String, #2 String
                # This matches the expected line splitting: line[0]=Quality, line[3]=#1 String, line[4]=#2 String
                df_export.to_csv(output_path, sep='\t', index=False, header=False, 
                                 columns=['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String'])

            print(f"Saved {split_name}.tsv")

        print(f"MRPC data successfully downloaded and saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_and_save_mrpc() 