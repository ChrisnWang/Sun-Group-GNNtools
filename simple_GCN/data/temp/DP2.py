import pandas as pd
import numpy as np

network = pd.read_csv("edges.csv")

network_len = len(network)
network1 = np.array(network)
for i in range(network_len):
    if  network1[i][0]>153:
        num = i - 1
        print(num)
        break
network = network.head(num)
print(network)
network_len1 = len(network)
network2 = np.array(network)
for i in range(network_len1):
    if network2[i][1] > 153:
        print(i)
        network = network.drop(index= i)
print(network)
new_data = network.loc[:, ~network.columns.str.contains("^Unnamed")]
new_data.to_csv("DP_edges.csv", index=False)

