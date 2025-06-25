Installation
============

## Prerequisites

* Python 3.9+
* Git
* `pip` (Python package installer)

## Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/michaWorku/amharic-ecommerce-data-extractor.git](https://github.com/michaWorku/amharic-ecommerce-data-extractor.git)
    cd amharic-ecommerce-data-extractor
    ```
    *(If you created the project in your current directory, you can skip ``git clone`` and ``cd``.)*

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a ``.env`` file in the root directory of your project. This file will store sensitive information like Telegram API keys.
    ```
    # .env
    TELEGRAM_API_ID="YOUR_TELEGRAM_API_ID"
    TELEGRAM_API_HASH="YOUR_TELEGRAM_API_HASH"
    # Add any other sensitive configurations here
    ```
    You can obtain your Telegram API ID and API Hash from `my.telegram.org <https://my.telegram.org/>`_.

5.  **Prepare Channels List:**
    Create a file named ``channels_to_crawl.txt`` inside the ``data/config/`` directory (e.g., ``data/config/channels_to_crawl.txt``). List the Telegram channel usernames (e.g., ``@Shageronlinestore``, ``@EthiopianMarketPlace``) you wish to scrape, one per line. The scraping script will read from this file.
    ```
    # config/channels_to_crawl.txt
    channel_username
    Shageronlinestore
    EthiopianMarketPlace
    Ethio_Mereja
    ```