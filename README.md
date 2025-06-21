# **Amharic-Ecommerce-Data-Extractor**

## **Project Description**

This project aims to transform unstructured text data from Amharic e-commerce Telegram channels into structured, machine-readable information. By leveraging advanced Natural Language Processing (NLP) techniques, specifically Named Entity Recognition (NER) with Large Language Models (LLMs), the system extracts key business entities such as product names, prices, and locations. This structured data is then used to populate a centralized database and, critically, to develop a "Vendor Scorecard" for FinTech micro-lending assessment.

The ultimate goal is to provide a comprehensive view of vendor activity and engagement, enabling data-driven decisions for potential loan offerings to active and promising e-commerce businesses on Telegram.

Key Features:

- **Telegram Data Ingestion:** Programmatic scraping of messages (text, images, documents) from selected Amharic e-commerce Telegram channels.
- **Amharic Text Preprocessing:** Robust cleaning, tokenization, and normalization tailored for Amharic linguistic features.
- **Named Entity Recognition (NER):** Fine-tuning transformer-based LLMs (e.g., XLM-Roberta, mBERT) to accurately identify `Product`, `Price`, and `Location` entities.
- **Model Comparison & Selection:** Evaluation of multiple NER models based on performance metrics (F1-score), speed, and robustness.
- **Model Interpretability:** Application of SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to understand model predictions and build trust.
- **Vendor Performance Scorecard Generation:** Combining extracted entities with Telegram post metadata (e.g., views, timestamps) to calculate key performance metrics (posting frequency, average views per post, average price point) and derive a weighted "Lending Score" for each vendor.

## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## **Installation**

### **Prerequisites**

- Python 3.9+
- Git
- `pip` (Python package installer)

### **Steps**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/your-username/amharic-ecommerce-extractor.git
    cd amharic-ecommerce-extractor
    
    ```
    
    If you created the project in the current directory:
    
    ```
    # Already in the project root
    
    ```
    
2. **Create and activate a virtual environment:**
    
    ```
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    
4. Configure Environment Variables:
    
    Create a .env file in the root directory of your project. This file will store sensitive information like Telegram API keys.
    
    ```
    # .env
    TELEGRAM_API_ID="YOUR_TELEGRAM_API_ID"
    TELEGRAM_API_HASH="YOUR_TELEGRAM_API_HASH"
    # Add any other sensitive configurations here
    
    ```
    
    You can obtain your Telegram API ID and API Hash from [my.telegram.org](https://my.telegram.org/).
    

## **Usage**

This project provides scripts to streamline the data pipeline, from ingestion to vendor scorecard generation.

### **1. Data Ingestion**

To start collecting data from Telegram channels:

```
python scripts/run_pipeline.py --stage ingest_data

```

*(Note: implement the `telegram_scraper.py` and integrate it into `run_pipeline.py`.)*

### **2. Data Labeling**

After data ingestion, you'll need to label a subset of your Amharic text data in CoNLL format. Refer to the `data/processed/` directory for where to save your labeled files.

### **3. NER Model Fine-tuning**

To fine-tune the NER model:

```
python scripts/run_pipeline.py --stage fine_tune_ner

```

(Note: This will execute the training process defined in src/models/ner_trainer.py.)

Explore the notebooks/ directory for detailed experimentation and prototyping steps for model training and interpretability.

### **4. Vendor Scorecard Generation**

Once the NER model is trained and entities are extracted, generate the vendor scorecards:

```
python scripts/run_pipeline.py --stage generate_scorecards

```

*(Note: This will utilize the logic in `src/analytics/vendor_scorecard.py`.)*

## **Project Structure**

```
├── .vscode/                 # VSCode specific settings for Python development
│   └── settings.json
├── .github/                 # GitHub specific configurations
│   └── workflows/
│       └── unittests.yml    # CI/CD workflow for tests and linting
├── .gitignore               # Specifies intentionally untracked files to ignore
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Modern Python packaging configuration (PEP 517/621)
├── README.md                # Project overview, installation, usage
├── Makefile                 # Common development tasks (setup, test, lint, clean)
├── .env                     # Environment variables (e.g., API keys - kept out of Git)
├── src/                     # Core source code for the project
│   ├── __init__.py          # Marks src as a Python package
│   ├── data_ingestion/      # Scripts for scraping and initial data collection
│   │   ├── __init__.py
│   │   └── telegram_scraper.py
│   ├── data_preprocessing/  # Modules for cleaning and transforming raw Amharic text
│   │   ├── __init__.py
│   │   └── text_processor.py
│   ├── models/          # Code for NER model fine-tuning, evaluation, and interpretability
│   │   ├── __init__.py
│   │   ├── ner_trainer.py
│   │   └── model_evaluator.py
│   ├── analytics/           # Logic for generating vendor performance metrics and scorecards
│   │   ├── __init__.py
│   │   └── vendor_scorecard.py
│   └── utils/               # General utility functions, custom tokenizers, CoNLL parsers
│       ├── __init__.py
│       ├── custom_tokenizers.py
│       └── conll_parser.py
├── tests/                   # Test suite (unit, integration)
│   ├── unit/                # Unit tests for individual components
│   │   ├── __init__.py
│   │   ├── test_data_ingestion.py
│   │   ├── test_data_preprocessing.py
│   │   └── test_ner_models.py
│   └── integration/         # Integration tests for combined components
│       ├── __init__.py
│       └── test_end_to_end.py
├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
│   ├── 01_EDA_and_Scraping_Prototyping.ipynb
│   ├── 02_NER_Fine_Tuning_Experimentation.ipynb
│   ├── 03_Model_Interpretability_Exploration.ipynb
│   ├── 04_Vendor_Scorecard_Development.ipynb
│   └── README.md
├── scripts/                 # Standalone utility scripts (e.g., data processing, deployment)
│   ├── run_pipeline.py
│   ├── generate_labels.py
│   └── README.md
├── docs/                    # Project documentation (e.g., Sphinx docs)
│   └── README.md
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data
│   ├── processed/           # Transformed, cleaned, or feature-engineered data
│   └── README.md
├── config/                  # Configuration files
│   ├── __init__.py
│   └── settings.py
└── examples/                # Example usage of the project components
    └── README.md

```

## **Development and Testing**

- **Running Tests:**
    
    ```
    make test
    # or
    pytest tests/
    
    ```
    
- **Linting:**
    
    ```
    make lint
    
    ```
    

## **Contributing**

Contributions are welcome! Please feel free to open issues or submit pull requests.

## **License**

This project is licensed under the [MIT License].