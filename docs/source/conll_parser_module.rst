CoNLL Parser Module (`src/utils/conll_parser.py`)
==================================================

This module provides helper functions for reading from and writing to CoNLL-formatted text files, which are commonly used for sequence labeling tasks like Named Entity Recognition.

**Key Responsibilities:**

* **Reading CoNLL Files:** Parses CoNLL files (token and label separated by whitespace) into a structured list of sentences, where each sentence is a list of dictionaries.
* **Writing CoNLL Files:** Writes structured data back into the CoNLL format, suitable for dataset creation or manual labeling tools.

**Example Usage:**

.. code-block:: python

    from src.utils.conll_parser import read_conll, write_conll

    # Example read
    data = read_conll('path/to/your/file.conll')

    # Example write
    sample_data = [[{'text': 'Hello', 'label': 'O'}, {'text': 'world', 'label': 'O'}]]
    write_conll(sample_data, 'output.conll')

.. currentmodule:: src.utils.conll_parser
.. automodule:: src.utils.conll_parser
   :members:
   :undoc-members:
   :show-inheritance: