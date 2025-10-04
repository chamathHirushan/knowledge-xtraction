import pandas as pd
from datasets import load_dataset

def load_pubmedqa(split="train"):
    """
    Loads and concatenates PubMedQA datasets (unlabeled, labeled, artificial),
    cleans the 'context' field, and returns a combined DataFrame.
    """
    # Load parquet files from HuggingFace Hub
    ds = load_dataset("riiwang/MESAQA", split=split)
    df = pd.DataFrame(ds)
    df = pd.read_parquet("hf://datasets/riiwang/MESAQA/train/train-00000-of-00001.parquet")
    # df2 = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_labeled/train-00000-of-00001.parquet")
    # df3 = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_artificial/train-00000-of-00001.parquet")

    # Concatenate all datasets
    #df = pd.concat([df1, df2, df3], ignore_index=True)

    # Extract the first context string from dicts
    df["context_clean"] = df["context"].apply(
        lambda x: x["contexts"][0] if isinstance(x, dict) and "contexts" in x else None
    )

    return df
