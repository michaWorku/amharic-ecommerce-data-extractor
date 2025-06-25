Vendor Scorecard Development
=============================

The Vendor Scorecard (`src/analytics/vendor_scorecard.py`) focuses on translating raw data and extracted entities into actionable business intelligence. This is the culmination of the data extraction process, directly supporting EthioMart's micro-lending objectives.

## Methodology:

1.  **Data Integration:** Merges preprocessed Telegram message data with the results of NER inference.
2.  **Entity Extraction (Inference):** The fine-tuned ``xlm-roberta-base`` model is used to extract ``PRODUCT``, ``PRICE``, ``LOCATION``, and ``CONTACT_INFO`` from all preprocessed messages.
3.  **Vendor Analytics Engine:** Calculates key performance indicators (KPIs) for each unique vendor (identified by ``channel_username``):
    * **Posting Frequency:** The total number of messages posted by a vendor.
    * **Average Views per Post:** Measures content engagement (sum of views / total posts).
    * **Posts per Week:** Calculated from the posting frequency over the observed period.
    * **Average Price Point:** Calculated by averaging all extracted numeric price entities.
    * **Top Product and Price:** Identifying the most prominent product and its associated price, potentially from the highest-viewed post.
4.  **Lending Score Formulation:** A composite "Lending Score" is formulated by normalizing selected performance indicators (e.g., Min-Max Scaling for Views, Posting Frequency) and applying predefined weights. This provides a single, interpretable metric to rank vendors based on their digital activity and potential.

## Example Vendor Scorecard Metrics:

.. list-table::
   :widths: 15 10 15 10 15 15 15 10 15
   :header-rows: 1

   * - Channel Username
     - Total Posts
     - Avg Views/Post
     - Posts/Week
     - Avg Price (ETB)
     - Top Product
     - Top Price (ETB)
     - Lending Score
   * - ``EthioMarketPlace``
     - 150
     - 2500
     - 15
     - 850
     - "T-shirt"
     - 300
     - 0.85
   * - ``shega_shop_1``
     - 120
     - 1800
     - 12
     - 1200
     - "Smartwatch"
     - 2000
     - 0.78
   * - ``Ethio_online_store``
     - 80
     - 3200
     - 8
     - 500
     - "Shoes"
     - 500
     - 0.70
   * - ``LocalDeals``
     - 200
     - 1500
     - 20
     - 700
     - "Books"
     - 150
     - 0.65
   * - ``GadgetHub_ET``
     - 90
     - 2800
     - 9
     - 4500
     - "Earbuds"
     - 1500
     - 0.60
*(These are illustrative figures. Your actual scorecard will vary based on scraped data.)*

## Business Impact:

The Vendor Scorecard directly supports EthioMart's business objectives by:
* **Quantifying Vendor Potential:** Moving beyond subjective assessment to data-driven insights for micro-lending.
* **Risk Reduction:** Identifying active and engaged vendors with consistent product offerings and clear contact information.
* **Optimized Lending Decisions:** Prioritizing vendors with higher "Lending Scores," indicating stronger online presence and transactional clarity.
* **Market Insight:** Understanding product trends, common price points, and active selling locations from the aggregated data.