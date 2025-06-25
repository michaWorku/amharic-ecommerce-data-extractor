Model Evaluator Module (`src/models/model_evaluator.py`)
==========================================================

This module handles the evaluation of the fine-tuned NER model and provides tools for model interpretability.

**Key Responsibilities:**

* **Model Loading:** Loads a fine-tuned NER model and its tokenizer from a local directory.
* **Entity Prediction:** Performs NER inference on new, unseen preprocessed text data.
* **Prediction Saving:** Saves the predicted entities (token-level labels) to a CSV file, formatted for review and potential manual correction.
* **Model Interpretability:** Integrates SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to provide insights into how the model makes its predictions, highlighting important words or features.

**Example Usage (via pipeline):**

.. code-block:: bash

    python scripts/run_pipeline.py --stage evaluate_ner

.. currentmodule:: src.models.model_evaluator
.. autoclass:: src.models.model_evaluator.NERModelEvaluator
   :members:
   :undoc-members:
   :show-inheritance: