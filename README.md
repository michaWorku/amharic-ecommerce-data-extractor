# **Amharic-Ecommerce-Data-Extractor**

## **Project Description**

This project delivers a robust FinTech solution designed to transform unstructured e-commerce text data from Amharic Telegram channels into structured, machine-readable information. By leveraging advanced Natural Language Processing (NLP) techniques, particularly fine-tuned Named Entity Recognition (NER) models based on Large Language Models (LLMs), the system accurately extracts key business entities such as **product names, prices, locations, and contact information**. This structured data is then used to populate a centralized database and, critically, to develop a comprehensive "Vendor Scorecard" for FinTech micro-lending assessment.

The ultimate goal is to provide EthioMart with a data-driven, quantifiable view of vendor activity, engagement, and market standing. This enables informed decision-making for potential loan offerings to active and promising e-commerce businesses operating on Telegram, thereby reducing lending risk and optimizing financial support.

**Key Features:**

- **Telegram Data Ingestion:** Programmatic and efficient scraping of raw message data (text and associated metadata like views, dates, channel information) from selected Amharic e-commerce Telegram channels. Includes detailed summary statistics for scraped data quality and volume per channel.
- **Amharic Text Preprocessing:** A modular and robust pipeline for cleaning, normalizing, and standardizing raw Amharic text data. This includes Unicode normalization, character and numeral mapping, removal of noise (URLs, mentions, hashtags, emojis), punctuation standardization, whitespace normalization, and the generation of a dedicated `tokens` column for downstream NLP tasks. Provides comprehensive summary statistics on preprocessing effectiveness and data completeness.
- **Named Entity Recognition (NER):** Fine-tuning state-of-the-art transformer-based LLMs (e.g., XLM-RoBERTa, mBERT) to accurately identify `Product`, `Price`, `Location`, and `Contact Info` entities in Amharic e-commerce messages.
- **Model Comparison & Selection:** Rigorous evaluation and comparison of multiple fine-tuned NER models based on performance metrics (F1-score, Precision, Recall), demonstrating `xlm-roberta-base` as the superior choice for this specific task.
- **Vendor Performance Scorecard Generation:** A sophisticated analytics engine that combines extracted entities with Telegram post metadata (views, timestamps, channel/vendor name) to calculate key performance indicators (KPIs) such as posting frequency, average views per post, average price point, and top-performing products. These KPIs are then aggregated into a weighted **"Lending Score"** for each vendor, providing actionable insights for micro-lending decisions.
- **Model Interpretability (Exploration):** Preliminary application of SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to understand feature contributions and local predictions of the NER model, building trust and transparency.

## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development and Testing](#development-and-testing)
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
    git clone https://github.com/michaWorku/amharic-ecommerce-data-extractor.git
    cd amharic-ecommerce-data-extractor
    
    ```
    
    *(If you created the project in your current directory, you can skip `git clone` and `cd`.)*
    
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
    
5. Prepare Channels List:
    
    Create a file named channels_to_crawl.txt inside the data/config/ directory (e.g., data/config/channels_to_crawl.txt). List the Telegram channel usernames (e.g., @Shageronlinestore, @EthiopianMarketPlace) you wish to scrape, one per line. The scraping script will read from this file.
    
    ```
    # data/config/channels_to_crawl.txt
    channel_username
    Shageronlinestore
    EthiopianMarketPlace
    Ethio_Mereja
    
    ```
    

## **Usage**

This project provides a streamlined pipeline via `scripts/run_pipeline.py` to execute each stage of the data processing.

### **1. Data Ingestion (Scraping Telegram Channels)**

Collect raw messages and metadata from Telegram channels. The scraper skips media downloads by default for efficiency.

```
python scripts/run_pipeline.py --stage ingest_data --limit 100 # Adjust limit as needed, e.g., 10000 for larger data

```

*(After execution, check `data/raw/telegram_data.csv` and the console output for scraping summary statistics.)*

### **2. Data Preprocessing**

Clean, normalize, and transform the raw Amharic text data. This generates `preprocessed_text` and `tokens` columns.

```
python scripts/run_pipeline.py --stage preprocess_data

```

*(After execution, check `data/processed/preprocessed_telegram_data.csv` and the console output for preprocessing summary statistics.)*

### **3. Named Entity Recognition (NER) Model Fine-tuning**

This stage involves preparing the labeled dataset and fine-tuning multilingual transformer models for Amharic NER.

- **Manual Data Labeling:** A subset of your `preprocessed_telegram_data.csv` (or similar cleaned text) needs to be manually labeled in CoNLL format for `Product`, `Price`, `Location`, and `Contact Info` entities. Refer to the `data/labeled/` directory for example formats.
- **Run Fine-tuning:**
    
    ```
    python scripts/run_pipeline.py --stage fine_tune_ner
    
    ```
    
    *(This executes the training process defined in `src/models/ner_trainer.py` and saves the fine-tuned model. Refer to `notebooks/02_Amharic_NER_Fine_Tuning_Experimentation.ipynb` for detailed experimentation and model comparison results.)*
    

### **4. Vendor Scorecard Generation**

Generate the comprehensive vendor scorecards by combining NER extracted entities with Telegram post metadata.

```
python scripts/run_pipeline.py --stage generate_scorecards

```

*(This utilizes the logic in `src/analytics/vendor_scorecard.py`. Refer to `notebooks/03_Vendor_Scorecard_Development.ipynb` for detailed methodology and results.)*

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
│   ├── models/              # Code for NER model fine-tuning, evaluation, and interpretability
│   │   ├── __init__.py
│   │   ├── ner_trainer.py
│   │   └── model_evaluator.py
│   ├── analytics/           # Logic for generating vendor performance metrics and scorecards
│   │   ├── __init__.py
│   │   └── vendor_scorecard.py
│   └── data_labeling/       # Module for CoNLL format labeling, CoNLL parsers
│       ├── __init__.py
│       ├── prepare_dat_for_labeling.py
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
│   ├── 02_Amharic_NER_Fine_Tuning_Experimentation.ipynb
│   ├── 03_Vendor_Scorecard_Development.ipynb
│   └── README.md            # README for notebooks directory
├── scripts/                 # Standalone utility scripts (e.g., pipeline orchestration, labeling helpers)
│   ├── run_pipeline.py
│   ├── generate_labels.py   # Placeholder/Helper for generating labeling input
│   └── README.md            # README for scripts directory
├── docs/                    # Project documentation (e.g., Sphinx docs)
│   └── README.md
├── data/                    # Data storage (raw, processed, labeled)
│   ├── raw/                 # Original, immutable raw data (e.g., telegram_data.csv)
│   ├── processed/           # Transformed, cleaned, or feature-engineered data (e.g., preprocessed_telegram_data.csv)
│   ├── labeled/             # Manually labeled datasets for NER training (e.g., 01_labeled_telegram_product_price_location.txt)
│   └── README.md
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── settings.py
│   └── channels_to_crawl.txt # List of Telegram channels to scrape
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