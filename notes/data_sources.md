# Data Sources and Links

The raw datasets used in this project are automatically downloaded and processed from the Hugging Face Hub (using the `prepare_data.py` script). Below you can find the links to these dataset sources and their purpose in the project.

## 1. SQID (Shopping Queries Image Dataset)
* **Source:** [crossingminds/shopping-queries-image-dataset](https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset)
* **Description:** Based on the Amazon ESCI dataset, this dataset contains visual (image) features (CLIP embeddings) for the products. It also contains text embeddings for the queries.
* **Downloaded Files:**
  * `query_features.parquet` (Query IDs and 768-dim CLIP text embeddings)
  * `product_image_urls.parquet` (Product IDs and product image URLs)
  * `product_features.parquet` (Product IDs, visual embeddings, and text embeddings)

## 2. Amazon ESCI Dataset
* **Source:** [khanmu2003/amazon-shopping-queries-dataset](https://huggingface.co/datasets/khanmu2003/amazon-shopping-queries-dataset)
* **Description:** A rich, official dataset containing search queries, matched products, and relevance labels for these matches (E: Exact, S: Substitute, etc.).
* **Downloaded Files:**
  * `shopping_queries_dataset_examples.parquet` (Raw query texts, product IDs, and ESCI Ground Truth labels)

## Processing Workflow
The `prepare_data.py` script:
1. Downloads the relevant parquet files from both sources (into the `data/raw/` directory).
2. Merges the visual/text vector data from SQID with the raw query texts and relevance labels from the ESCI dataset.
3. Saves the cleaned parquet files, ready for training and database (`SQID` database) ingestion, into the `data/processed/` directory.
