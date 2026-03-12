import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df["target"] = wine.target

df.to_csv("wine.csv", index=False)
print("CSV guardado como wine.csv")
print("Clases:", wine.target_names)
print("Distribución de clases:\n", df["target"].value_counts().sort_index())
