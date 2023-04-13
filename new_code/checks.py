import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

paths = [
    # "2023-04-07/16-28-23",
    # "2023-04-07/17-56-41",
    # "2023-04-07/18-31-18",

    # "2023-04-12/09-28-53",
    "2023-04-12/14-50-10",
    "2023-04-12/15-33-19",
    # "2023-04-12/16-58-31",
    # "2023-04-12/21-30-11",
    # "2023-04-12/22-33-41",
    #
    "2023-04-13/08-21-26",
    # "2023-04-13/09-12-20"
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

