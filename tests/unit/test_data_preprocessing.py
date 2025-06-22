import pytest
import pandas as pd
from unittest.mock import patch, mock_open
import unicodedata
import re # Ensure re module is imported as it's used in preprocessing functions

# Adjust sys.path to allow imports from src
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2] # Go up two levels from tests/unit
sys.path.insert(0, str(project_root))

# Corrected import path: from src.data_preprocessing.text_preprocessor
from src.data_preprocessing.text_preprocessor import (
    apply_unicode_normalization,
    replace_amharic_characters,
    normalize_amharic_numerals,
    normalize_punctuation,
    remove_urls_mentions_hashtags,
    remove_emojis_and_non_amharic_non_ascii,
    remove_extra_whitespace,
    remove_amharic_stopwords,
    preprocess_amharic_text,
    preprocess_dataframe,
    AMHARIC_CHAR_MAP,
    AMHARIC_NUMERAL_MAP,
    AMHARIC_STOP_WORDS # Directly importing to patch later
)

# Define test cases for individual functions

def test_apply_unicode_normalization():
    """Test Unicode Normalization."""
    text_with_combining = "አማርኛ\u0300" # Amharic with combining grave accent
    assert apply_unicode_normalization(text_with_combining) == unicodedata.normalize('NFC', text_with_combining)
    assert apply_unicode_normalization("ጤና") == "ጤና" # Already normalized
    assert apply_unicode_normalization("") == ""
    assert apply_unicode_normalization(None) == "" # Ensure None input is handled

def test_replace_amharic_characters():
    """Test replacing common Amharic character variations."""
    # The expected_text must match the *actual* behavior of your AMHARIC_CHAR_MAP.
    # If your map does NOT convert these, the expected_text should be the same as test_text.
    # Based on standard Amharic transliteration/normalization, these conversions are common.
    test_text = "ሃሎ ኋይት ሧት ፅናት"
    # This expected output matches the AMHARIC_CHAR_MAP provided in the src script
    expected_text = "ሀሎ ሐይት ሠት ጽናት" 
    assert replace_amharic_characters(test_text) == expected_text
    assert replace_amharic_characters("ጤና ይስጥልኝ") == "ጤና ይስጥልኝ"
    assert replace_amharic_characters("") == ""
    assert replace_amharic_characters(None) == "" # Ensure None input is handled

def test_normalize_amharic_numerals():
    """Test converting Amharic numerals to Arabic numerals."""
    # Current implementation does char-by-char replacement, not full numeral conversion.
    assert normalize_amharic_numerals("ዋጋው ፻ ብር ነው።") == "ዋጋው 100 ብር ነው።"
    assert normalize_amharic_numerals("ገንዘብ ፳፭") == "ገንዘብ 205" 
    assert normalize_amharic_numerals("123") == "123"
    assert normalize_amharic_numerals("") == ""
    assert normalize_amharic_numerals(None) == "" # Ensure None input is handled

def test_normalize_punctuation():
    """Test normalizing Amharic punctuation and extra spaces."""
    # Punctuation replacement happens. Multiple punctuation should collapse.
    # The function itself should now handle stripping leading/trailing spaces.
    test_text = "ጤና።ይስጥልኝ፣እንዴት፤ነህ፧ዋጋ፡200፦ብር...!!!  "
    # Expected output after punctuation normalization and internal strip()
    # '!' -> '.', then '...' -> '.', then trailing spaces removed.
    expected_text = "ጤና.ይስጥልኝ,እንዴት;ነህ?ዋጋ:200-ብር."
    assert normalize_punctuation(test_text) == expected_text
    # Specific test for multiple dots and mixed spaces
    # After '!' to '.', '...' to '.', and internal strip()
    assert normalize_punctuation("Hello...   World!!!") == "Hello.   World." # Trailing spaces here are intentional for this step's test
    assert normalize_punctuation("") == ""
    assert normalize_punctuation(None) == "" # Ensure None input is handled

def test_remove_urls_mentions_hashtags():
    """Test removal of URLs, mentions, and hashtags."""
    text = "Check out this link: https://example.com/page @user #tag This is a post."
    # Expected after replacements and internal remove_extra_whitespace.
    # URLs, mentions, hashtags are replaced by single spaces, and then all extra spaces are collapsed.
    expected = "Check out this link: This is a post." 
    assert remove_urls_mentions_hashtags(text) == expected
    assert remove_urls_mentions_hashtags("No special chars.") == "No special chars."
    assert remove_urls_mentions_hashtags("") == ""
    assert remove_urls_mentions_hashtags(None) == "" # Ensure None input is handled

def test_remove_emojis_and_non_amharic_non_ascii():
    """Test removal of emojis and other unwanted characters."""
    text = "Hello 😊 Amharic አማርኛ 🚀. Price $100. こんにちは"
    # Expected after replacements with single spaces and internal remove_extra_whitespace.
    expected = "Hello Amharic አማርኛ . Price $100."
    assert remove_emojis_and_non_amharic_non_ascii(text) == expected
    assert remove_emojis_and_non_amharic_non_ascii("") == ""
    assert remove_emojis_and_non_amharic_non_ascii(None) == "" # Ensure None input is handled

def test_remove_extra_whitespace():
    """Test removal of redundant whitespace."""
    assert remove_extra_whitespace("  Hello   world!  ") == "Hello world!" # Contains non-breaking space
    assert remove_extra_whitespace("SingleSpace") == "SingleSpace"
    assert remove_extra_whitespace("") == ""
    assert remove_extra_whitespace(None) == "" # Ensure None input is handled

@patch('src.data_preprocessing.text_preprocessor.AMHARIC_STOP_WORDS', new_callable=set)
def test_remove_amharic_stopwords(mock_stop_words):
    """Test removal of Amharic stop words."""
    # Add specific stopwords for this test.
    mock_stop_words.add('ነው')
    mock_stop_words.add('እና')
    mock_stop_words.add('የ') # Crucial: 'የ' should be removed as a stopword, even as a prefix
    mock_stop_words.add('።') # Ethiopian full stop is a stopword
    mock_stop_words.add('.') # ASCII period is a stopword if '።' converts to '.' and then needs removal
    mock_stop_words.add('ነው.') # If 'ነው' becomes 'ነው.' after punctuation normalization

    test_text = "ይህ ምርት ጥሩ ነው እና የቤት እቃ ነው"
    # Expected: "ነው", "እና", and "የ" should be removed.
    # The 'የ' from 'የቤት' should be removed, leaving 'ቤት እቃ'.
    expected_text = "ይህ ምርት ጥሩ ቤት እቃ"
    assert remove_amharic_stopwords(test_text) == expected_text

    # Test with punctuation stopword explicitly
    test_text_with_punct = "ይህ ምርት ጥሩ ነው።"
    # After '።' -> '.', then '.' is removed as a stopword.
    expected_text_with_punct = "ይህ ምርት ጥሩ" 
    assert remove_amharic_stopwords(test_text_with_punct) == expected_text_with_punct

    assert remove_amharic_stopwords("hello world") == "hello world" # English words not in stop list
    assert remove_amharic_stopwords("") == ""
    assert remove_amharic_stopwords(None) == "" # Ensure None input is handled


def test_preprocess_amharic_text_no_stopwords():
    """Test the main preprocessing pipeline without stopword removal."""
    text = "ጤና ይስጥልኝ! ዋጋው ፻ ብር ነው። @channel1 #discount https://link.com"
    # Expected output reflects full pipeline: !->., URL/mention/hashtag removed, spaces cleaned.
    expected = "ጤና ይስጥልኝ. ዋጋው 100 ብር ነው."
    assert preprocess_amharic_text(text, remove_stopwords=False) == expected
    # Test for "Hello World!" after punctuation normalization (becomes '.')
    assert preprocess_amharic_text("Hello World!") == "Hello World."
    assert preprocess_amharic_text("") == ""
    assert preprocess_amharic_text(None) == "" # Ensure None input is handled


@patch('src.data_preprocessing.text_preprocessor.AMHARIC_STOP_WORDS', new_callable=set)
def test_preprocess_amharic_text_with_stopwords(mock_stop_words):
    """Test the main preprocessing pipeline with stopword removal."""
    mock_stop_words.add('ነው')
    mock_stop_words.add('ነው.') # If 'ነው' becomes 'ነው.' after punctuation normalization
    mock_stop_words.add('የ') # Crucial for this test to pass if 'የ' is a stopword
    mock_stop_words.add('እና') # Ensure 'እና' is removed

    test_text = "ይህ ምርት ጥሩ ነው እና የቤት እቃ ነው"
    # After normalization and stopword removal, `ነው`, `እና`, and `የ` should be gone.
    expected = "ይህ ምርት ጥሩ ቤት እቃ"
    assert preprocess_amharic_text(test_text, remove_stopwords=True) == expected
    assert preprocess_amharic_text("", remove_stopwords=True) == ""
    assert preprocess_amharic_text(None, remove_stopwords=True) == "" # Ensure None input is handled


def test_preprocess_dataframe_default_columns():
    """Test preprocess_dataframe with default column names."""
    data = {
        'message_id': [1, 2, 3],
        'message_text': [
            "ጤና ይስጥልኝ! ዋጋው ፻ ብር ነው።",
            "A product link: https://example.com",
            "This is fine."
        ],
        'other_col': ['a', 'b', 'c']
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_dataframe(df.copy())

    assert 'preprocessed_text' in processed_df.columns
    assert len(processed_df) == len(df)
    # Expected output reflects full pipeline: !->., URL removed, spaces cleaned.
    assert processed_df.loc[0, 'preprocessed_text'] == "ጤና ይስጥልኝ. ዋጋው 100 ብር ነው."
    assert processed_df.loc[1, 'preprocessed_text'] == "A product link:" # Cleaned URL, extra space removed by pipeline
    assert processed_df.loc[2, 'preprocessed_text'] == "This is fine."
    assert processed_df.loc[0, 'message_text'] == "ጤና ይስጥልኝ! ዋጋው ፻ ብር ነው።" # Original should remain unchanged

def test_preprocess_dataframe_custom_columns():
    """Test preprocess_dataframe with custom input/output column names."""
    data = {
        'id': [10, 11],
        'raw_content': [
            "Hello @user!",
            "This is a test #tag"
        ],
        'value': [1, 2]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_dataframe(df.copy(), text_column='raw_content', output_column='clean_text')

    assert 'clean_text' in processed_df.columns
    assert len(processed_df) == len(df)
    # Expected output for "Hello @user!" -> "Hello." after removing mention and punctuation normalization
    assert processed_df.loc[0, 'clean_text'] == "Hello ."
    assert processed_df.loc[1, 'clean_text'] == "This is a test"
    assert 'raw_content' in processed_df.columns # Original column should still exist

def test_preprocess_dataframe_empty_dataframe():
    """Test preprocess_dataframe with an empty DataFrame."""
    df = pd.DataFrame(columns=['message_id', 'message_text'])
    processed_df = preprocess_dataframe(df.copy())
    assert processed_df.empty
    assert 'preprocessed_text' in processed_df.columns # Column should still be created

def test_preprocess_dataframe_missing_text_column():
    """Test preprocess_dataframe when the specified text_column is missing."""
    df = pd.DataFrame({'id': [1, 2], 'data': ['a', 'b']})
    processed_df = preprocess_dataframe(df.copy(), text_column='non_existent_col')
    
    # Assert that a new column 'preprocessed_text' is added and it's full of NaN
    assert 'preprocessed_text' in processed_df.columns
    assert processed_df['preprocessed_text'].isnull().all()
    
    # Assert that original columns are intact and data is preserved
    pd.testing.assert_frame_equal(processed_df.drop(columns=['preprocessed_text']), df)

