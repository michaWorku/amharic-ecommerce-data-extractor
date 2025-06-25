Data Preprocessing Module (`src/data_preprocessing/text_preprocessor.py`)
=========================================================================

The `src/data_preprocessing/text_preprocessor.py` module is central to cleaning and preparing raw Amharic text for downstream NLP tasks.

**Key Responsibilities:**

* **Unicode Normalization (NFC):** Standardizes character representation.
* **Amharic Character & Numeral Replacement:** Canonicalizes variant Amharic characters and converts Amharic numerals to Arabic equivalents.
* **Noise Removal:** Eliminates URLs, Telegram mentions (@username), hashtags (#tag), emojis, and non-Amharic/non-ASCII characters.
* **Punctuation Normalization:** Standardizes punctuation marks and collapses repetitions.
* **Whitespace Normalization:** Removes redundant spaces and trims leading/trailing spaces.
* **Tokenization:** Generates a `tokens` column containing a list of individual words/punctuation marks from the preprocessed text.
* **Robust NaN Handling:** Ensures robust handling of missing values in input dataframes.
* **Summary Statistics:** Provides insights into preprocessing effectiveness, including counts of empty preprocessed texts and token lists.

**Example Usage (via pipeline):**

.. code-block:: bash

    python scripts/run_pipeline.py --stage preprocess_data

.. currentmodule:: src.data_preprocessing.text_preprocessor
.. automodule:: src.data_preprocessing.text_preprocessor
   :members:
   :undoc-members:
   :show-inheritance: