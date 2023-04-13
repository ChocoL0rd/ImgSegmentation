import pandas as pd
import os
from omegaconf import OmegaConf


outputs_path = "../outputs"

data_dirs = {
    "2023-04-09": "all",
    "2023-04-10": "all",
    "2023-04-12": "all",
    "2023-04-13": "all",
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
        print(date, time)
        cfg = OmegaConf.load(os.path.join(outputs_path, date, time, ".hydra/config.yaml"))
        model = cfg.model_cfg.name
        if cfg.model_cfg.load_pretrained:
            pretrained_path = cfg.model_cfg.pretrained_path
        else:
            pretrained_path = None

        tmp_df = pd.read_excel(os.path.join(outputs_path, date, time, "validation/descr_results.xlsx"))
        tmp_df = tmp_df.rename(columns={"Unnamed: 0": "stat"})
        tmp_df = tmp_df[tmp_df["stat"] == "mean"].drop("stat", axis=1)
        tmp_df.loc[:, "model"] = model
        tmp_df.loc[:, "date_time"] = date + "/" + time
        tmp_df.loc[:, "pretrained_path"] = pretrained_path
        # tmp_df.loc[:, "time"] = time
        df = pd.concat([df, tmp_df])

df = df.sort_values("soft_dice")
df.to_excel(f"table.xlsx", index=False)


