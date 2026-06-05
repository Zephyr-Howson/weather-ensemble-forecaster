import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

print(df.shape)
print(df.columns.tolist())
print(df.head())

df[
    [
        "source_count__max_temp",
        "source_std__max_temp",
        "source_range__max_temp"
    ]
].describe()