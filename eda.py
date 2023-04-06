#%%
import pandas as pd
# %%
df = pd.read_csv('wine_quality.csv')
df.sample(10)
# %%
df = pd.read_csv('winequalityN.csv')
df.head(10)
# %%
from sklearn.preprocessing import LabelEncoder
labels_enc = LabelEncoder()
df['type'] = labels_enc.fit_transform(df['type'])
# %%
labels_enc.classes_
# %%
df.head(5)
# %%
mapping = dict(zip(labels_enc.classes_, range(len(labels_enc.classes_))))
# %%
mapping
# %%
print(len(df))
df = df.dropna()
print(len(df))
# %%
