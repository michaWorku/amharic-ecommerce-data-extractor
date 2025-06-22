import pytest
import os
from unittest.mock import mock_open, patch
from typing import List, Dict

# Adjust sys.path to allow imports from src
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2] # Go up two levels from tests/unit
sys.path.insert(0, str(project_root))

from src.data_labeling.conll_parser import read_conll, write_conll

# Sample CoNLL data for testing
SAMPLE_CONLL_CONTENT = """
Token1	O
Token2	B-PRODUCT
Token3	I-PRODUCT
Token4	O

TokenA	B-LOC
TokenB	I-LOC
TokenC	O
TokenD	B-PRICE
TokenE	I-PRICE

Another	O
Sentence	O
.	O
"""

# Expected parsed data from SAMPLE_CONLL_CONTENT
EXPECTED_PARSED_DATA = [
    [
        {'text': 'Token1', 'label': 'O'},
        {'text': 'Token2', 'label': 'B-PRODUCT'},
        {'text': 'Token3', 'label': 'I-PRODUCT'},
        {'text': 'Token4', 'label': 'O'}
    ],
    [
        {'text': 'TokenA', 'label': 'B-LOC'},
        {'text': 'TokenB', 'label': 'I-LOC'},
        {'text': 'TokenC', 'label': 'O'},
        {'text': 'TokenD', 'label': 'B-PRICE'},
        {'text': 'TokenE', 'label': 'I-PRICE'}
    ],
    [
        {'text': 'Another', 'label': 'O'},
        {'text': 'Sentence', 'label': 'O'},
        {'text': '.', 'label': 'O'}
    ]
]

def test_read_conll_success():
    """Test successful reading of a well-formatted CoNLL file."""
    with patch("builtins.open", mock_open(read_data=SAMPLE_CONLL_CONTENT)):
        with patch("os.path.exists", return_value=True):
            data = read_conll("dummy_path.txt")
            assert data == EXPECTED_PARSED_DATA

def test_read_conll_empty_file():
    """Test reading an empty CoNLL file."""
    with patch("builtins.open", mock_open(read_data="")):
        with patch("os.path.exists", return_value=True):
            data = read_conll("empty_file.txt")
            assert data == []

def test_read_conll_file_not_found():
    """Test reading a CoNLL file that does not exist."""
    with patch("os.path.exists", return_value=False):
        data = read_conll("non_existent_file.txt")
        assert data == [] # Should return empty list and print warning

def test_read_conll_malformed_line():
    """Test reading a CoNLL file with a malformed line."""
    malformed_content = "Token1 O\nMalformedLineWithNoTab\nToken3\tLABEL"
    with patch("builtins.open", mock_open(read_data=malformed_content)):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(ValueError, match="Malformed CoNLL line"):
                read_conll("malformed_file.txt")

def test_read_conll_trailing_blank_lines():
    """Test reading a CoNLL file that ends with multiple blank lines."""
    content_with_trailing_blanks = SAMPLE_CONLL_CONTENT + "\n\n\n"
    with patch("builtins.open", mock_open(read_data=content_with_trailing_blanks)):
        with patch("os.path.exists", return_value=True):
            data = read_conll("dummy_path_trailing_blanks.txt")
            assert data == EXPECTED_PARSED_DATA # Should not add empty sentences

def test_write_conll_success(tmp_path):
    """Test successful writing of data to a CoNLL file."""
    output_file = tmp_path / "output.txt"
    write_conll(EXPECTED_PARSED_DATA, str(output_file))
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Normalize expected content to match how write_conll formats it (always ends with a blank line after each sentence)
    expected_content_normalized = ""
    for sentence in EXPECTED_PARSED_DATA:
        for token_data in sentence:
            expected_content_normalized += f"{token_data['text']}\t{token_data['label']}\n"
        expected_content_normalized += "\n" # Add a blank line after each sentence

    assert content == expected_content_normalized

def test_write_conll_empty_data(tmp_path):
    """Test writing empty data to a CoNLL file."""
    output_file = tmp_path / "empty_output.txt"
    write_conll([], str(output_file))
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == "" # Empty data should result in an empty file

def test_write_conll_creates_directory(tmp_path):
    """Test that write_conll creates necessary directories."""
    nested_dir = tmp_path / "nested" / "output"
    output_file = nested_dir / "output.txt"
    
    # Ensure the directory does not exist initially
    assert not nested_dir.exists()
    
    write_conll(EXPECTED_PARSED_DATA, str(output_file))
    
    # Assert that the directory was created and the file exists
    assert nested_dir.is_dir()
    assert output_file.is_file()
    
    # Verify content
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    expected_content_normalized = ""
    for sentence in EXPECTED_PARSED_DATA:
        for token_data in sentence:
            expected_content_normalized += f"{token_data['text']}\t{token_data['label']}\n"
        expected_content_normalized += "\n"
    assert content == expected_content_normalized

