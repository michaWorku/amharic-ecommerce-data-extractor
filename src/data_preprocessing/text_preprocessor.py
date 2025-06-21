import re
import unicodedata
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)

# --- Amharic Character Mappings (from references) ---
# Map for common Amharic character variations to a canonical form
# This mapping needs to be comprehensive based on observed data.
# This is a representative sample based on typical Amharic preprocessing needs.
AMHARIC_CHAR_MAP = {"ሐ":"ሀ","ሑ":"ሁ","ሒ":"ሂ","ሓ":"ሃ","ሔ":"ሄ","ሕ":"ህ","ሖ":"ሆ",\
                       "ኀ":"ሀ","ኁ":"ሁ","ኂ":"ሂ","ኃ":"ሃ","ኄ":"ሄ","ኅ":"ህ","ኆ":"ሆ",\
                       "ሠ":"ሰ","ሡ":"ሱ","ሢ":"ሲ","ሣ":"ሳ","ሤ":"ሴ","ሥ":"ስ","ሦ":"ሶ","ሧ":"ሷ",\
                       "ዐ":"አ","ዑ":"ኡ","ዒ":"ኢ","ዓ":"ኣ","ዔ":"ኤ","ዕ":"እ","ዖ":"ኦ",\
                       "ጸ":"ፀ","ጹ":"ፁ","ጺ":"ፂ","ጻ":"ፃ","ጼ":"ፄ","ጽ":"ፅ","ጾ":"ፆ"}
# AMHARIC_CHAR_MAP = {
#     'ሃ': 'ሀ', 'ሁ': 'ሁ', 'ሂ': 'ሂ', 'ሄ': 'ሄ', 'ህ': 'ህ', 'ሆ': 'ሆ', # ሀ variants
#     'ሏ': 'ላ', 'ሉ': 'ሉ', 'ሊ': 'ሊ', 'ሌ': 'ሌ', 'ል': 'ል', 'ሎ': 'ሎ', # ለ variants
#     'ኋ': 'ሐ', 'ሗ': 'ሐ', 'ሑ': 'ሑ', 'ሒ': 'ሒ', 'ሔ': 'ሔ', 'ሕ': 'ሕ', 'ሖ': 'ሖ', # ሐ variants
#     'ሟ': 'ማ', 'ሙ': 'ሙ', 'ሚ': 'ሚ', 'ሜ': 'ሜ', 'ም': 'ም', 'ሞ': 'ሞ', # መ variants
#     'ሧ': 'ሠ', 'ሡ': 'ሠ', 'ሢ': 'ሠ', 'ሤ': 'ሠ', 'ሥ': 'ሠ', 'ሦ': 'ሠ', # ሠ variants
#     'ሯ': 'ራ', 'ሩ': 'ሩ', 'ሪ': 'ሪ', 'ሬ': 'ሬ', 'ር': 'ር', 'ሮ': 'ሮ', # ረ variants
#     'ሷ': 'ሳ', 'ሱ': 'ሱ', 'ሲ': 'ሲ', 'ሴ': 'ሴ', 'ስ': 'ስ', 'ሶ': 'ሶ', # ሰ variants
#     'ሿ': 'ሻ', 'ሹ': 'ሹ', 'ሺ': 'ሺ', 'ሼ': 'ሼ', 'ሽ': 'ሽ', 'ሾ': 'ሾ', # ሸ variants
#     'ዷ': 'ዳ', 'ዱ': 'ዱ', 'ዲ': 'ዲ', 'ዴ': 'ዴ', 'ድ': 'ድ', 'ዶ': 'ዶ', # ደ variants
#     'ጇ': 'ጀ', 'ጁ': 'ጁ', 'ጂ': 'ጂ', 'ጄ': 'ጄ', 'ጅ': 'ጅ', 'ጆ': 'ጆ', # ጀ variants
#     'ጧ': 'ጣ', 'ጡ': 'ጡ', 'ጢ': 'ጢ', 'ጤ': 'ጤ', 'ጥ': 'ጥ', 'ጦ': 'ጦ', # ጠ variants
#     'ፏ': 'ፋ', 'ፉ': 'ፉ', 'ፊ': 'ፊ', 'ፌ': 'ፌ', 'ፍ': 'ፍ', 'ፎ': 'ፎ', # ፈ variants
#     'ኗ': 'ና', 'ኙ': 'ኙ', 'ኚ': 'ኚ', 'ኜ': 'ኜ', 'ኝ': 'ኝ', 'ኞ': 'ኞ', # ኘ variants
#     'ኋ': 'ወ', # Another form of ኋ
#     'ዐ': 'አ', 'ዓ': 'አ', # Different forms of 'Ayn' to 'Alef'
#     'ፅ': 'ጽ', 'ፀ': 'ጸ', # Ethiopian Tsadi (ጸ) and Tsadi (ፅ)
#     'ሏ': 'ለ', # Another form of ሏ
#     'ጨ': 'ጠ', # Example: sometimes ጩ is used instead of ጥ
#     'ቸ': 'ከ',
#     'ቯ': 'በ', # Vee sound in some keyboards, normalize to Be
# }

# Amharic Numerals to Arabic Numerals mapping
AMHARIC_NUMERAL_MAP = {
    '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
    '፮': '6', '፯': '7', '፰': '8', '፱': '9', # ፰፻ represents 100 in Ethiopian system (one hundred)
    # Note: Amharic numbers are often written with combinations, e.g., ፲ (10), ፳ (20), ፴ (30) ... ፼ (10,000)
    # For simplicity, we convert single digit Geez numerals. More complex conversion might be needed
    # if compound Geez numbers are common in your data.
    '፲': '10', '፳': '20', '፴': '30', '፵': '40', '፶': '50',
    '፷': '60', '፸': '70', '፹': '80', '፺': '90', '፻': '100', '፼': '10000'
}


def apply_unicode_normalization(text: str) -> str:
    """Applies Unicode Normalization Form C (NFC)."""
    return unicodedata.normalize('NFC', text)

def replace_amharic_characters(text: str) -> str:
    """Replaces common Amharic character variations with a canonical form."""
    for amh_char, canonical_char in AMHARIC_CHAR_MAP.items():
        text = text.replace(amh_char, canonical_char)
    return text

def normalize_amharic_numerals(text: str) -> str:
    """Converts Amharic numerals to Arabic numerals."""
    for amh_num, arabic_num in AMHARIC_NUMERAL_MAP.items():
        text = text.replace(amh_num, arabic_num)
    return text

def normalize_punctuation(text: str) -> str:
    """
    Normalizes Amharic punctuation characters to their ASCII equivalents.
    Removes other non-essential symbols.
    """
    text = text.replace('።', '.')  # Ethiopian full stop
    text = text.replace('፣', ',')  # Ethiopian comma
    text = text.replace('፤', ';')  # Ethiopian semicolon
    text = text.replace('፧', '?')  # Ethiopian question mark
    text = text.replace('፡', ':')  # Ethiopian colon
    text = text.replace('፦', '-')  # Ethiopian dash/hyphen (used for e.g. price)

    # Replace multiple consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_urls_mentions_hashtags(text: str) -> str:
    """Removes URLs, Telegram mentions (@username), and hashtags (#tag)."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
    text = re.sub(r'@\w+', '', text) # Mentions
    text = re.sub(r'#\w+', '', text) # Hashtags
    return text

def remove_emojis_and_non_amharic_non_ascii(text: str) -> str:
    """
    Removes emojis and characters that are not Amharic script, basic ASCII (English letters, numbers,
    common punctuation), or whitespace.
    """
    # Define a regex pattern that matches:
    # 1. Amharic script characters (\u1200-\u137F)
    # 2. Basic Latin characters (a-zA-Z)
    # 3. Digits (0-9)
    # 4. Basic punctuation and symbols often needed (.,!?;:()[]{}'\"-/)
    # 5. Whitespace (\s)
    # Characters outside this set will be removed.
    pattern = re.compile(r'[^\u1200-\u137F\u0020-\u007E\s]+')
    text = pattern.sub('', text)
    return text

def remove_extra_whitespace(text: str) -> str:
    """Replaces multiple spaces with a single space and strips leading/trailing whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

# --- Stop Word Removal (Placeholder) ---
# A comprehensive Amharic stop word list is crucial.
# You might need to build one or find a reliable source.
# For now, this is a placeholder.
AMHARIC_STOP_WORDS = set([
    # 'ነው', 'እና', 'የ', 'በ', 'ለ', 'ከ', 'ላይ', 'ውስጥ', 'እንደ', 'ም', 'ማለት', 'አለ', 'አይደለም'
    # Add common Amharic stop words here
    # Example: 'ነው', 'እና', 'የ', 'በ', 'ለ', 'ከ', 'ላይ', 'ውስጥ', 'እንደ', 'ም', 'ማለት', 'አለ', 'አይደለም'
    # For a real project, this list would be much larger and potentially loaded from a file.
])

def remove_amharic_stopwords(text: str) -> str:
    """
    Removes Amharic stop words from the text.
    Requires a predefined list of stop words.
    """
    if not AMHARIC_STOP_WORDS:
        logger.warning("Amharic stop words list is empty. Stop word removal will not be effective.")
        return text
    
    words = text.split()
    filtered_words = [word for word in words if word not in AMHARIC_STOP_WORDS]
    return " ".join(filtered_words)

# --- Main Preprocessing Function ---
def preprocess_amharic_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Applies a series of robust preprocessing steps to a single Amharic text string.

    Args:
        text (str): The input Amharic text string.
        remove_stopwords (bool): Whether to remove common Amharic stop words.
                                 Set to True if AMHARIC_STOP_WORDS is populated.

    Returns:
        str: The preprocessed text string.
    """
    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully

    text = apply_unicode_normalization(text)
    text = replace_amharic_characters(text)
    text = normalize_amharic_numerals(text) # Apply this before removing non-ASCII if numbers are important
    text = remove_urls_mentions_hashtags(text)
    text = remove_emojis_and_non_amharic_non_ascii(text) # Removes unwanted characters
    text = normalize_punctuation(text) # Normalize specific punctuation then remove extra spaces
    text = remove_extra_whitespace(text)

    if remove_stopwords:
        text = remove_amharic_stopwords(text)
        text = remove_extra_whitespace(text) # Clean up spaces after stop word removal

    return text

def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'message_text', output_column: str = 'preprocessed_text', remove_stopwords: bool = False) -> pd.DataFrame:
    """
    Applies the comprehensive Amharic preprocessing pipeline to a specified text column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the raw text messages.
        output_column (str): The name of the new column to store the preprocessed text.
        remove_stopwords (bool): Whether to apply stop word removal during preprocessing.

    Returns:
        pd.DataFrame: A new DataFrame with the preprocessed text column added.
    """
    if text_column not in df.columns:
        logger.error(f"DataFrame must contain a '{text_column}' column for preprocessing.")
        return df

    logger.info(f"Starting text preprocessing on column '{text_column}'...")
    
    # Fill NaN values in the text column with empty strings to prevent errors
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].fillna('')

    df_copy[output_column] = df_copy[text_column].apply(lambda x: preprocess_amharic_text(x, remove_stopwords=remove_stopwords))
    
    logger.info("Text preprocessing complete.")
    return df_copy

if __name__ == '__main__':
    # --- Example Usage for individual functions ---
    print("--- Individual Function Tests ---")
    sample_text_complex = "ጤና ይስጥልኝ! ዋጋው ፻፳፭ ብር ነው። አድራሻችን መገናኛ ስሪ ኤም ሲቲ ሞል ነው። @Shageronlinestore #ቅናሽ 😊 https://t.me/example_product"
    print(f"Original: {sample_text_complex}")

    step1 = apply_unicode_normalization(sample_text_complex)
    print(f"1. Unicode Norm: {step1}")

    step2 = replace_amharic_characters(step1)
    print(f"2. Char Replace: {step2}")

    step3 = normalize_amharic_numerals(step2)
    print(f"3. Numeral Norm: {step3}")

    step4 = remove_urls_mentions_hashtags(step3)
    print(f"4. Removed URLs/Mentions/Hashtags: {step4}")

    step5 = remove_emojis_and_non_amharic_non_ascii(step4)
    print(f"5. Removed Emojis/Non-Amharic: {step5}")

    step6 = normalize_punctuation(step5)
    print(f"6. Punctuation Norm: {step6}")

    step7 = remove_extra_whitespace(step6)
    print(f"7. Whitespace Norm: {step7}")

    final_processed = preprocess_amharic_text(sample_text_complex)
    print(f"\nFinal Preprocessed (no stopwords): {final_processed}")

    # --- Example Usage for DataFrame processing ---
    print("\n--- DataFrame Preprocessing Test ---")
    dummy_data = {
        'message_id': [1, 2, 3, 4, 5],
        'message_text': [
            "ጤና ይስጥልኝ! ዋጋው 500 ብር ነው። አድራሻችን መገናኛ ስሪ ኤም ሲቲ ሞል ነው። @Shageronlinestore #ቅናሽ",
            "ይህ ምርት በጣም ቆንጆ ነው። ዋጋ፦ 1,200 ብር. https://t.me/example_product",
            "አዲስ እቃ ገብቷል!!! ውስን ፍሬ ነው ያለው።",
            "Hello, this is a test message. Some ዎርዶች and numbers like ፳፫፬፭.", # Mixed text & Geez numerals
            "፩ ፪ ፫ ፬ ፭ ፮ ፯ ፰ ፱ ፲ ፳ ፻። የዋጋ ቅናሽ አለ።" # Pure Geez numerals
        ],
        'message_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'views': [100, 150, 80, 120, 90],
        'media_path': [None, 'photos/ch1_msg2.jpg', None, None, None]
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    print("\nOriginal DataFrame Sample:")
    print(dummy_df[['message_id', 'message_text']].head())

    processed_df = preprocess_dataframe(dummy_df.copy())
    print("\nProcessed DataFrame Sample:")
    print(processed_df[['message_id', 'message_text', 'preprocessed_text']].head())

    # Example with stopword removal (assuming AMHARIC_STOP_WORDS is populated)
    # For demonstration, let's temporarily populate it
    AMHARIC_STOP_WORDS.update(['ነው', 'የ', 'እና', 'በ', 'ለ', 'ከ'])
    processed_df_with_stopwords = preprocess_dataframe(dummy_df.copy(), remove_stopwords=True)
    print("\nProcessed DataFrame with Stopwords Removed Sample:")
    print(processed_df_with_stopwords[['message_id', 'message_text', 'preprocessed_text']].head())
    
    # Clean up the temporary stop word addition
    AMHARIC_STOP_WORDS.clear()

    # Verify structural integrity (example of saving)
    # Note: Adjust path if running from main project root for testing purposes
    # For actual pipeline execution, run_pipeline.py handles the path
    output_test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed', 'processed_telegram_data_test.csv')
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    processed_df.to_csv(output_test_path, index=False, encoding='utf-8')
    logger.info(f"Test processed data saved to {output_test_path}")

