import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

paths = [
    "2023-04-07/16-28-23",
    "2023-04-07/17-56-41",
    "2023-04-07/18-31-18"
]

df = {}

for path in paths:
    tmp_df = pd.read_excel(os.path.join("outputs", path, "validation/full_results.xlsx"))
    df[path] = tmp_df["soft_jaccard"]
    imgs = tmp_df["img"].tolist()

df = pd.DataFrame(df)
imgs = [i[:-5] for i in imgs]

# corr_matrix = df.corr()
# sn.heatmap(corr_matrix, annot=True)

plt.xticks(range(len(imgs)), imgs, rotation='vertical')
plt.plot(df, label=df.columns)
plt.legend()

# x = [0, len(imgs)-1]
# for i in df.columns:
#     plt.plot(x, [df[i].mean(), df[i].mean()], label=i)
# plt.legend()

plt.show()

