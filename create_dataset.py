import pandas as pd
from datasets import load_dataset


def create_merged_dataset():
    print("Loading data from Hugging Face...")

    # 1. Load the ESCI dataset (query, product ID and labels)
    print("Loading ESCI dataset...")
    esci_dataset = load_dataset(
        "khanmu2003/amazon-shopping-queries-dataset", data_files="*examples.parquet"
    )
    df_esci = esci_dataset["train"].to_pandas()

    # 2. Load the product image links (product ID and image URLs)
    print("Loading image dataset...")
    images_dataset = load_dataset(
        "crossingminds/shopping-queries-image-dataset", data_files="data/product_image_urls.parquet"
    )
    df_images = images_dataset["train"].to_pandas()

    print("Merging datasets...")
    merged_df = pd.merge(df_esci, df_images, on="product_id", how="left")

    print("Building the DataFrame in the target format...")
    # 'query' -> user_query
    # 'product_title' -> product
    # 'image_url' -> imagelink
    # 'product_id' -> product_id
    # 'esci_label' -> label

    df_final = pd.DataFrame(
        {
            "user_query": merged_df["query"],
            "product": merged_df.get("product_title", pd.Series(dtype=str)),
            "product_id": merged_df["product_id"],
            "imagelink": merged_df.get(
                "image_url", merged_df.get("product_image_url", pd.Series(dtype=str))
            ),  # fallback for safety
            "label": merged_df["esci_label"],
        }
    )

    print("Merge complete! First 5 rows:")
    print(df_final.head())

    return df_final


if __name__ == "__main__":
    df = create_merged_dataset()
    print("Saving to CSV: merged_dataset.csv")
    df.to_csv("merged_dataset.csv", index=False)
    print("Done.")
