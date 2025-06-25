import os
import re
from typing import List, Dict

def read_conll(file_path: str) -> List[List[Dict[str, str]]]:
    """
    Reads a CoNLL formatted file and parses it into a list of sentences.
    Each sentence is a list of dictionaries, where each dictionary represents
    a token and its associated label. It is now more flexible with whitespace
    delimiters.

    Args:
        file_path (str): The path to the CoNLL formatted text file.

    Returns:
        List[List[Dict[str, str]]]: A list of sentences, where each sentence
                                     is a list of {'text': token, 'label': label} dictionaries.
                                     Returns an empty list if the file is empty or not found.
    Raises:
        ValueError: If a line in the CoNLL file does not contain exactly two parts
                    (token and label) after splitting by whitespace.
    """
    if not os.path.exists(file_path):
        print(f"Warning: CoNLL file not found at {file_path}. Returning empty list.")
        return []

    sentences: List[List[Dict[str, str]]] = []
    current_sentence: List[Dict[str, str]] = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip() # Remove leading/trailing whitespace including newlines
            if not line:  # Blank line indicates end of a sentence
                if current_sentence:  # Only add if the sentence is not empty
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                # Use re.split to split by one or more whitespace characters
                # This is more robust than line.split('\t')
                parts = re.split(r'\s+', line)
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed CoNLL line at {file_path}:{line_num}. "
                        f"Expected 'token\tlabel' or 'token  label' (any whitespace delimiter), got '{line}'"
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
    into a CoNLL formatted text file, using a tab ('\t') as a delimiter.

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
                # Always write with a tab to maintain consistent output format
                f.write(f"{token_data['text']}\t{token_data['label']}\n")
            f.write("\n") # Blank line to separate sentences

if __name__ == '__main__':
    # Simple demonstration of read_conll and write_conll
    sample_data = [
        [{'text': 'This', 'label': 'O'}, {'text': 'is', 'label': 'O'}, {'text': 'a', 'label': 'O'}, {'text': 'test', 'label': 'O'}, {'text': 'sentence', 'label': 'O'}, {'text': '.', 'label': 'O'}],
        [{'text': 'Addis', 'label': 'B-LOC'}, {'text': 'Ababa', 'label': 'I-LOC'}, {'text': 'has', 'label': 'O'}, {'text': 'many', 'label': 'O'}, {'text': 'products', 'label': 'B-PRODUCT'}, {'text': '.', 'label': 'O'}]
    ]
    test_output_path = 'test_output.conll'
    write_conll(sample_data, test_output_path)
    print(f"Sample data written to {test_output_path}")

    read_data = read_conll(test_output_path)
    print("\nRead data:")
    for sentence in read_data:
        for token_data in sentence:
            print(f"{token_data['text']}\t{token_data['label']}")
        print()

    # Clean up the test file
    os.remove(test_output_path)
    print(f"Cleaned up {test_output_path}")
