Usage (Pipeline Stages)
========================

This project provides a streamlined pipeline via ``scripts/run_pipeline.py`` to execute each stage of the data processing. Make sure you are in the project's root directory when running these commands and your Python virtual environment is activated.

---

### **1. Data Ingestion (Scraping Telegram Channels)**

Collect raw messages and metadata from Telegram channels. The scraper skips media downloads by default for efficiency.

.. code-block:: bash

    python scripts/run_pipeline.py --stage ingest_data --limit 1000 # Adjust limit as needed, e.g., 100 for quick testing

*(After execution, check ``data/raw/telegram_data.csv`` and the console output for scraping summary statistics.)*

---

### **2. Data Preprocessing**

Clean, normalize, and transform the raw Amharic text data. This generates ``preprocessed_text`` and ``tokens`` columns.

.. code-block:: bash

    python scripts/run_pipeline.py --stage preprocess_data

*(After execution, check ``data/processed/preprocessed_telegram_data.csv`` and the console output for preprocessing summary statistics.)*

---

### **3. Named Entity Recognition (NER) Model Fine-tuning**

This stage involves preparing the labeled dataset and fine-tuning multilingual transformer models for Amharic NER.

* **Manual Data Labeling:** A subset of your ``preprocessed_telegram_data.csv`` (or similar cleaned text) needs to be manually labeled in CoNLL format for ``Product``, ``Price``, ``Location``, and ``Contact Info`` entities. Refer to the ``data/labeled/`` directory for example formats.
    *You need to create your labeled data file at ``data/labeled/01_labeled_telegram_product_price_location.txt`` for this stage to run successfully.*
* **Run Fine-tuning:**
    .. code-block:: bash

        python scripts/run_pipeline.py --stage fine_tune_ner --model_name xlm-roberta-base --epochs 3 --batch_size 16

    *(This executes the training process defined in ``src/models/ner_trainer.py`` and saves the fine-tuned model to ``models/fine_tuned_ner_model``. Refer to ``notebooks/02_Amharic_NER_Fine_Tuning_Experimentation.ipynb`` for detailed experimentation and model comparison results.)*

---

### **4. NER Model Evaluation & Interpretability**

Evaluate the fine-tuned NER model on your labeled data and perform interpretability analysis to understand its predictions.

.. code-block:: bash

    python scripts/run_pipeline.py --stage evaluate_ner

*(This uses ``src/models/model_evaluator.py``. It saves predicted labels to ``data/predicted/telegram_predictions.csv`` and prints interpretability insights to the console. For full interactive visualizations of SHAP/LIME, refer to the ``02_Amharic_NER_Fine_Tuning_Experimentation.ipynb`` notebook.)*

---

### **5. Vendor Scorecard Generation**

Generate the comprehensive vendor scorecards by combining NER extracted entities with Telegram post metadata.

.. code-block:: bash

    python scripts/run_pipeline.py --stage generate_scorecards

*(This utilizes the logic in ``src/analytics/vendor_scorecard.py``. After execution, check ``data/analytics/vendor_scorecard.csv`` for the output. Refer to ``notebooks/03_Vendor_Scorecard_Development.ipynb`` for detailed methodology and results visualization.)*

---

### **6. Run All Stages**

You can run all the stages sequentially:

.. code-block:: bash

    python scripts/run_pipeline.py --stage all --limit 1000 # Adjust limit for ingestion

*(Note: Running ``all`` will also trigger ``fine_tune_ner`` and ``evaluate_ner``, which require labeled data and can be time/resource-intensive.)*