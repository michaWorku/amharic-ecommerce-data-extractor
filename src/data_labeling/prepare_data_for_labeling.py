import pandas as pd
import os
import sys
from pathlib import Path

# Adjust sys.path to allow imports from src
project_root = Path(__file__).resolve().parents[2] # Adjust if script is not in tests/unit
sys.path.insert(0, str(project_root))

from src.data_preprocessing.text_preprocessor import preprocess_amharic_text

def prepare_data_for_manual_labeling(
    input_csv_path: str = 'data/raw/telegram_data.csv',
    output_txt_path: str = 'data/labeled/messages_for_labeling.txt',
    num_messages: int = 50,
    text_column: str = 'Message',
    use_preprocessed_text: bool = True # Set to False if you want to label raw text
) -> None:
    """
    Loads text messages, preprocesses them into tokens, and saves them
    to a text file, ready for manual CoNLL labeling.

    Args:
        input_csv_path (str): Path to the input CSV file containing messages.
        output_txt_path (str): Path to the output text file for labeling.
        num_messages (int): The number of messages to prepare for labeling.
        text_column (str): The name of the column containing the text messages.
        use_preprocessed_text (bool): If True, uses the preprocessed text from
                                     'preprocessed_text' column if available,
                                     otherwise preprocesses the 'Message' column.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if df.empty:
        print("Input DataFrame is empty. No messages to label.")
        return

    messages_to_process = []
    # Determine which column to use for labeling
    if use_preprocessed_text and 'preprocessed_text' in df.columns:
        print(f"Using 'preprocessed_text' column for labeling from {input_csv_path}.")
        messages_to_process = df['preprocessed_text'].dropna().tolist()
    elif text_column in df.columns:
        print(f"Using '{text_column}' column for labeling from {input_csv_path}.")
        messages_to_process = df[text_column].dropna().tolist()
    else:
        print(f"Error: Neither 'preprocessed_text' nor '{text_column}' column found in the CSV.")
        return

    # Limit to the requested number of messages
    messages_to_process = messages_to_process[:num_messages]

    output_dir = os.path.dirname(output_txt_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        print(f"Preparing {len(messages_to_process)} messages for labeling. Output to {output_txt_path}")
        for i, message_text in enumerate(messages_to_process):
            # Preprocess the text into tokens.
            # return_tokens=True makes it return a list of tokens directly.
            tokens = preprocess_amharic_text(message_text, remove_stopwords=False, return_tokens=True)
            
            for token in tokens:
                # Write token and a placeholder 'O' for you to manually change
                f.write(f"{token}\tO\n") # Using tab for separation might be easier for some editors
            f.write("\n") # Blank line to separate messages

    print(f"Successfully prepared messages for manual labeling in '{output_txt_path}'.")
    print(f"Please open this file and replace 'O' with appropriate CoNLL labels (B-PRODUCT, I-PRODUCT, B-LOC, etc.).")
    print("Remember to separate sentences/messages with a blank line.")

# ... (rest of the script remains the same) ...

if __name__ == "__main__":
    # Change input_csv_path to point to your preprocessed data
    prepare_data_for_manual_labeling(
        input_csv_path='data/processed/preprocessed_telegram_data.csv',
        output_txt_path='data/labeled/messages_for_labeling.txt',
        num_messages=50, 
        text_column='Message', # This column exists in raw, but we're now using preprocessed_text
        use_preprocessed_text=True # Keep this as True if you want to label the cleaned text
    )

    # # Use the raw data, and explicitly tell it NOT to look for 'preprocessed_text'
    # prepare_data_for_manual_labeling(
    #     input_csv_path='data/raw/telegram_data.csv', # <--- Keep this line
    #     output_txt_path='data/labeled/messages_for_labeling.txt',
    #     num_messages=50, 
    #     text_column='Message', # <--- Ensure this points to your raw message column
    #     use_preprocessed_text=False # <--- CHANGE THIS LINE to False
    # )