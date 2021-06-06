import pandas as pd
from matplotlib import pyplot as plt

# "date";"sample";"width";"length";"thick"
files = pd.read_csv("raw_data/mechaniczne_02.csv", sep=";")
files["length"] = files["length"].str.replace(',', '.').astype(float)
files["width"] = files["width"].str.replace(',', '.').astype(float)
files["thick"] = files["thick"].str.replace(',', '.').astype(float)

for index, row in files.iterrows():
    filename = row["date"] + "_" + str(row["sample"])
    width = row["width"] / 1000
    length = row["length"]
    thick = row["thick"] / 1000
    A = width * thick

    tmp = pd.read_csv(f"raw_data/zrywanie/{filename}.txt", sep=",", skiprows=5)
    tmp = tmp.drop([0])
    # "Load (N)","Time (s)","Extension (mm)","Stress (MPa)","Strain (mm/mm)"
    tmp = tmp[tmp["Load (N)"] > 0]
    tmp = tmp[tmp["Extension (mm)"] > 0]
    tmp = tmp[:tmp[["Load (N)"]].idxmax()[0]]
    tmp["tension"] = 0
    tmp["strain"] = 0

    tmp["tension"] = tmp["Load (N)"] / A
    tmp["strain"] = tmp["Extension (mm)"] / length
    plt.plot(tmp["strain"], tmp["tension"])
    plt.show()



