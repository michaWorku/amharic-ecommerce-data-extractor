import os
from typing import List, Dict
import sys
from pathlib import Path

def read_conll(file_path: str) -> List[List[Dict[str, str]]]:
    """
    Reads a CoNLL formatted file and parses it into a list of sentences.
    Each sentence is a list of dictionaries, where each dictionary represents
    a token and its associated label.

    Args:
        file_path (str): The path to the CoNLL formatted text file.

    Returns:
        List[List[Dict[str, str]]]: A list of sentences, where each sentence
                                     is a list of {'text': token, 'label': label} dictionaries.
                                     Returns an empty list if the file is empty or not found.
    Raises:
        ValueError: If a line in the CoNLL file does not contain exactly two parts
                    (token and label).
    """
    if not os.path.exists(file_path):
        print(f"Warning: CoNLL file not found at {file_path}. Returning empty list.")
        return []

    sentences: List[List[Dict[str, str]]] = []
    current_sentence: List[Dict[str, str]] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Blank line indicates end of a sentence
                if current_sentence:  # Only add if the sentence is not empty
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t') # CoNLL format typically uses tab separation
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed CoNLL line at {file_path}:{line_num}. "
                        f"Expected 'token\\tlabel', got '{line}'"
                    )
                token, label = parts
                current_sentence.append({'text': token, 'label': label})
        
        # Add the last sentence if the file doesn't end with a blank line
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

def write_conll(data: List[List[Dict[str, str]]], file_path: str) -> None:
    """
    Writes structured data (list of sentences, each with tokens and labels)
    into a CoNLL formatted text file.

    Args:
        data (List[List[Dict[str, str]]]): The data to write, in the format
                                            [[{'text': token, 'label': label}, ...], ...].
        file_path (str): The path to the output CoNLL formatted text file.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in data:
            for token_data in sentence:
                f.write(f"{token_data['text']}\t{token_data['label']}\n")
            f.write("\n") # Blank line to separate sentences


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2] # Adjust if script is not in tests/unit
    sys.path.insert(0, str(project_root))
    from src.data_labeling.conll_parser import read_conll

    labeled_data = read_conll('data/labeled/labeled_telegram_product_price_location.txt')

    # You can inspect the first sentence and its tokens/labels:
    # print(labeled_data[0])