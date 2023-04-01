import pandas as pd
import os
from omegaconf import OmegaConf


outputs_path = "../outputs"

data_dirs = {
    # "2023-03-09": "all",
    # "2023-03-11": "all",
    # "2023-03-13": "all",
    # "2023-03-14": "all",
    # "2023-03-15": "all",
    # "2023-03-18": "all",
    # "2023-03-19": "all",
    # "2023-03-20": "all",
    # "2023-03-21": "all",
    # "2023-03-23": "all",
    # "2023-03-24": "all",
    # "2023-03-25": "all",
    "2023-03-26": "all",
    "2023-03-27": "all",
    "2023-03-28": "all",
}


df = pd.DataFrame()

for date in data_dirs:
    if data_dirs[date] == "all":
        time_list = os.listdir(os.path.join(outputs_path, date))
    elif type(data_dirs[date]) == list:
        time_list = data_dirs[date]
    else:
        raise Exception("keys of data_dirs is a list or all")

    for time in time_list:
        cfg = OmegaConf.load(os.path.join(outputs_path, date, time, ".hydra/config.yaml"))
        model = cfg.model_conf.name
        if cfg.load_pretrained:
            pretrained_path = cfg.pretrained_path
        else:
            pretrained_path = None

        tmp_df = pd.read_excel(os.path.join(outputs_path, date, time, "validation/top_results.xlsx"))
        tmp_df.loc[:, "model"] = model
        tmp_df.loc[:, "date_time"] = date + "/" + time
        tmp_df.loc[:, "pretrained_path"] = pretrained_path
        # tmp_df.loc[:, "time"] = time
        df = pd.concat([df, tmp_df])

for metric_name in df.metric.unique():
    for top in df.top.unique():
        tmp_df = df[(df.metric == metric_name) & (df.top == top)].sort_values(["result"])
        tmp_df.to_excel(f"{metric_name}_{top}_table.xlsx", index=False)
#
# df[df.metric == "dice"].sort_values(["result"]).to_excel("dice_table.xlsx", index=False)
# df[df.metric == "jaccard"].sort_values(["result"]).to_excel("jaccard_table.xlsx", index=False)


