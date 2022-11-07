###########################
# TOP COMMANDS
###########################
# create empty session
globals().clear()

# load libraries
import os
import pandas as pd
import numpy as np

# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

# https://medium.com/@lucapetriconi/churn-modeling-a-detailed-step-by-step-guide-in-python-1e96d51c7523
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/

# https://towardsdatascience.com/survival-analysis-in-python-a-model-for-customer-churn-e737c5242822
# https://www.kaggle.com/code/alessandromarceddu/churn-survival-analysis

###########################
# LOAD DATA
###########################

df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))

###########################
# Descriptives
###########################

df.head()
df.info()
df["Churn Label"].value_counts()
