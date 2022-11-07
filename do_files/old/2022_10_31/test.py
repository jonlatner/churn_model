globals().clear()

import pandas as pd
import numpy as np

import seaborn as sns
import missingno as msno

import matplotlib.pyplot as plt

import os

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

# Load 
data_raw = pd.read_csv('https://raw.githubusercontent.com/lucamarcelo/Churn-Modeling/main/Customer%20Churn%20Data.csv')

fig = plt.gcf()
sns.pairplot(data = data_raw, 
                hue='Churn'
    )
fig.set_size_inches(w=4, h=8, forward=True)
# plt.show()
plt.savefig(os.path.join(main_dir,graphs,"test_300dpi.pdf"))

