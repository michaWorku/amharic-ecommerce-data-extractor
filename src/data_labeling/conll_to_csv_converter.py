import pandas as pd
import os
import sys
from pathlib import Path
from typing import List, Dict

# Adjust sys.path to allow imports from src
project_root = Path(__file__).resolve().parents[2] # Go up two levels from src/data_labeling
sys.path.insert(0, str(project_root))

from src.data_labeling.conll_parser import read_conll

def convert_conll_to_csv(
    conll_file_path: str = 'data/labeled/labeled_telegram_product_price_location.txt',
    output_csv_path: str = 'data/processed/labeled_data.csv'
) -> None:
    """
    Converts a CoNLL formatted text file into a CSV file with 'message_text'
    and 'labels_sequence' columns.

    Args:
        conll_file_path (str): Path to the input CoNLL formatted text file.
        output_csv_path (str): Path to the output CSV file.
    """
    print(f"Reading CoNLL data from: {conll_file_path}")
    conll_data = read_conll(conll_file_path)

    if not conll_data:
        print(f"No data found in {conll_file_path}. Exiting.")
        return

    messages = []
    labels_sequences = []

    for sentence_data in conll_data:
        tokens = [item['text'] for item in sentence_data]
        labels = [item['label'] for item in sentence_data]
        
        messages.append(' '.join(tokens))
        labels_sequences.append(' '.join(labels))

    df = pd.DataFrame({
        'message_text': messages,
        'labels_sequence': labels_sequences
    })

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully converted CoNLL data to CSV: {output_csv_path}")

if __name__ == "__main__":
    convert_conll_to_csv()
