Contributing
============

Contributions are welcome! Please feel free to open issues or submit pull requests.

## Setup for Development

1.  Fork the repository.
2.  Clone your forked repository: ``git clone https://github.com/YOUR_USERNAME/amharic-ecommerce-data-extractor.git``
3.  Navigate to the project directory: ``cd amharic-ecommerce-data-extractor``
4.  Create and activate a virtual environment:
    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate

5.  Install development dependencies:
    .. code-block:: bash

        pip install -r requirements.txt
        pip install -r requirements-dev.txt # If you have a separate dev requirements file

## Running Tests

.. code-block:: bash

    make test
    # or
    pytest tests/

## Linting

.. code-block:: bash

    make lint

## Building Documentation Locally

To build the Sphinx documentation locally to preview your changes:

.. code-block:: bash

    cd docs
    make html
    # Then open build/html/index.html in your browser