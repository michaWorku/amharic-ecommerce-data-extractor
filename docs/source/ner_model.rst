NER Model Fine-Tuning and Evaluation
====================================

The core of our solution for extracting business entities lies in a fine-tuned NER model. This section details the process of preparing a labeled dataset, selecting a base model, and fine-tuning it for our specific Amharic e-commerce entities, followed by a discussion of the evaluation results.

## Labeled Dataset Preparation

The NER model requires a dataset where specific entities (Product, Price, Location, Contact Info) are manually labeled. This project utilized a CoNLL-formatted dataset (`data/labeled/01_labeled_telegram_product_price_location.txt`). The labels follow the B-I-O (Beginning, Inside, Outside) tagging scheme.

**Detected Labels:**

* ``B-CONTACT_INFO``
* ``I-CONTACT_INFO``
* ``B-LOC``
* ``I-LOC``
* ``B-PRICE``
* ``I-PRICE``
* ``B-PRODUCT``
* ``I-PRODUCT``
* ``O`` (Outside of any entity)

The dataset was split into 80% for training and 20% for evaluation.

* **Total Samples:** 3257 sentences (Example)
* **Train Samples:** 2605 sentences (Example)
* **Eval Samples:** 652 sentences (Example)

## Model Selection and Fine-Tuning

We compared the performance of three multilingual transformer models, widely used for their capabilities in diverse languages, on our Amharic NER task:

1.  ``xlm-roberta-base``: A powerful multilingual model.
2.  ``distilbert-base-multilingual-cased``: A smaller, faster version of BERT, suitable for multilingual tasks.
3.  ``bert-base-multilingual-cased`` (mBERT): A foundational multilingual BERT model.

Each model was fine-tuned for 3 epochs with a learning rate of 2e-5 and a batch size of 16. The evaluation metrics (Precision, Recall, F1-score) were computed using ``seqeval``, focusing on a macro average to account for potential class imbalance.

**Evaluation Results Comparison:**

.. list-table::
   :widths: 25 15 15 15 15
   :header-rows: 1

   * - Model
     - Eval Loss
     - Eval Precision
     - Eval Recall
     - Eval F1-Score
   * - ``xlm-roberta-base``
     - 0.0229
     - 0.9702
     - 0.9789
     - **0.9745**
   * - ``distilbert-base-multilingual-cased``
     - 0.0706
     - 0.8951
     - 0.8693
     - 0.8816
   * - ``bert-base-multilingual-cased``
     - 0.0598
     - 0.8967
     - 0.9117
     - 0.9038

**Analysis of Model Performance:**

* ``xlm-roberta-base`` emerged as the clear winner, achieving the highest F1-score of 0.9745. This indicates superior performance in correctly identifying and classifying entities in Amharic e-commerce messages. Its higher precision and recall values across the board demonstrate its robustness.
* ``bert-base-multilingual-cased`` (mBERT) performed commendably, with an F1-score of 0.9038.
* ``distilbert-base-multilingual-cased`` showed the lowest performance among the three, with an F1-score of 0.8816, though still a respectable result.

**Per-Label Metrics (Selected highlights for ``xlm-roberta-base``):**

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - Label
     - Precision
     - Recall
     - F1-Score
   * - ``CONTACT_INFO``
     - 0.99
     - 1.00
     - 1.00
   * - ``LOC``
     - 0.98
     - 0.99
     - 0.98
   * - ``PRICE``
     - 0.98
     - 0.98
     - 0.98
   * - ``PRODUCT``
     - 0.93
     - 0.94
     - 0.94

``xlm-roberta-base`` consistently performs exceptionally well across all entity types. While ``CONTACT_INFO`` is almost perfectly identified by all models (likely due to its structured nature - phone numbers, etc.), ``xlm-roberta-base`` maintains a significant edge in extracting more variable entities like ``LOC``, ``PRICE``, and ``PRODUCT``. This detailed per-label analysis solidifies ``xlm-roberta-base`` as the optimal choice for this task, balancing high accuracy across all critical entities.

## Selected Model for Production: ``xlm-roberta-base``

Based on the robust evaluation, ``xlm-roberta-base`` was selected as the best-performing model due to its superior F1-score across all evaluated entities. While larger in size compared to DistilBERT, its accuracy is critical for providing reliable insights for micro-lending decisions. The fine-tuned model has been saved locally (``./models/fine_tuned_ner_model``) and can be pushed to the Hugging Face Hub for easy access and deployment.