Data Ingestion Module (`src/data_ingestion/telegram_scraper.py`)
===================================================================

The `src/data_ingestion/telegram_scraper.py` module is responsible for programmatically collecting raw message data from specified Telegram channels.

**Key Responsibilities:**

* **Channel Reading:** Reads Telegram channel usernames from `config/channels_to_crawl.txt`.
* **Message Scraping:** Connects to Telegram using `telethon` and scrapes messages, including text, date, views, and sender information.
* **Data Persistence:** Appends new messages to `data/raw/telegram_data.csv`.
* **Metadata Capture:** Extracts and stores essential metadata like `channel_title`, `channel_username`, `message_id`, `message_date`, `views`, `media_path`, and `media_type`.
* **Summary Statistics:** Provides detailed statistics about the scraped data, including total messages, unique channels, and missing values.

**Example Usage (via pipeline):**

.. code-block:: bash

    python scripts/run_pipeline.py --stage ingest_data --limit 500

.. currentmodule:: src.data_ingestion.telegram_scraper
.. automodule:: src.data_ingestion.telegram_scraper
   :members:
   :undoc-members:
   :show-inheritance: