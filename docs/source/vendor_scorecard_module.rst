Vendor Scorecard Module (`src/analytics/vendor_scorecard.py`)
===============================================================

This module implements the core logic for generating the FinTech Vendor Scorecard, which is crucial for micro-lending assessment.

**Key Responsibilities:**

* **Data Integration:** Merges NER predicted entities with original Telegram post metadata (views, dates, channel information).
* **KPI Calculation:** Computes key performance indicators (KPIs) for each unique vendor (identified by `channel_username`). These KPIs include:
    * Total Posts
    * Average Views per Post
    * Posts per Week
    * Average Price Point (from extracted prices)
    * Top Product and its Price (from highest-viewed posts)
* **Lending Score Formulation:** Normalizes selected KPIs and applies predefined weights to derive a composite "Lending Score," providing a quantifiable metric for vendor ranking and financial decision-making.

**Example Usage (via pipeline):**

.. code-block:: bash

    python scripts/run_pipeline.py --stage generate_scorecards

.. currentmodule:: src.analytics.vendor_scorecard
.. autoclass:: src.analytics.vendor_scorecard.VendorScorecardGenerator
   :members:
   :undoc-members:
   :show-inheritance: