import pandas as pd

def format_context(context_list):
    if isinstance(context_list, list):
        return ". ".join(context_list)
    return str(context_list)

def load_mesaqa():
    df = pd.read_json("hf://datasets/riiwang/MESAQA/MultiSpanQA.json")

    df.drop(columns=["evidence","evidence_idx"], inplace=True)
    df["context"] = df["context"].apply(format_context)

    return df