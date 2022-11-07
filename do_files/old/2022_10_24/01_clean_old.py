###########################
# TOP COMMANDS
###########################
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np
import janitor # clean column names

## Visualization
from plotnine import *
        
# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

# https://medium.com/@lucapetriconi/churn-modeling-a-detailed-step-by-step-guide-in-python-1e96d51c7523
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/

###########################
# LOAD DATA
###########################

df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))

###########################
# Define dictionary with the key-value pair to remap.
###########################

dict_category = {
"customerid":"id",
"lat_long":"geography",
"city":"geography",
"churn_reason":"dv",
"payment_method":"service",
"tech_support":"service",
"online_security":"service",
"internet_service":"service",
"streaming_tv":"service",
"streaming_movies":"service",
"online_backup":"service",
"device_protection":"service",
"contract":"service",
"multiple_lines":"service",
"senior_citizen":"demographics",
"phone_service":"service",
"dependents":"demographics",
"churn_label":"dv",
"paperless_billing":"service",
"partner":"demographics",
"gender":"demographics",
"state":"geography",
"country":"geography",
"zip_code":"geography",
"latitude":"geography",
"longitude":"geography",
"tenure_months":"duration",
"monthly_charges":"charges",
"total_charges":"charges"
}

dict_value = {
        "No internet service" : 'N/A',
        "No phone service" : 'N/A',
        "Bank transfer (automatic)" : 'Transfer',
        "Credit card (automatic)": ' Credit card',
        "Electronic check" : ' Check (E)',
        "Mailed check": 'Check (M)',
        "Month-to-month": 'M-to-m'
        }

###########################
# Missings
###########################

# Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)

# clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

# missing values?
df_missing=pd.DataFrame(df.isnull().sum()).reset_index()
df_missing.columns =['variable', 'count']
df_missing

df_graph = (ggplot(df_missing, aes(y='count', x='reorder(variable, -count)', label = "count"))
+ geom_col(stat="identity")
+ geom_text(size=10, ha="left")
+ coord_flip()
+ ylab("Count")
+ theme(
        axis_title_y = element_blank(),
)
)
ggsave(plot = df_graph, filename = "graph_missing.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)
del(df_missing,df_graph)

###########################
# Create table for structure of the data
###########################

# select non-numeric columns and count unique factors
df_cat = df.select_dtypes(exclude=[np.number])
df_cat["count"]=1
df_cat = df_cat.melt(id_vars=["count"]).drop_duplicates()
df_cat=pd.DataFrame(df_cat.groupby(["variable"])["count"].sum()).sort_values(by=['count']).reset_index()
df_cat["type"]="factor"
df_cat

# select numeric columns
df_num = df.select_dtypes(include=[np.number])
df_num["count"]=1
df_num = df_num.melt(id_vars=["count"])
df_num = df_num.drop(columns=["value"]).drop_duplicates().reset_index(drop=True)
df_num["type"]="numeric"

# combine numeric and non-numeric columns
df_variables = pd.concat([df_cat,df_num]).reset_index(drop=True)
df_variables["number"] = np.arange(len(df_variables))+1
df_variables["category"] = df_variables["variable"]

df_variables=df_variables.replace({"category": dict_category})
df_variables

df_cat_keep2["value"].value_counts(dropna=False)


df_variables=df_variables[["number","variable","type","count"]]

# create table
df_variables.to_latex(os.path.join(main_dir,tables,"table_variables.tex"),index=False)
del(df_num,df_cat,df_variables)

###########################
# Graph churn
###########################

df_cat = df.select_dtypes(exclude=[np.number])
df_cat["count"]=1

# rotate
df_cat_keep = df_cat.melt(id_vars=["count"])
df_cat_keep = pd.DataFrame(df_cat_keep.groupby(["variable","value"])["count"].sum()).sort_values(by=['count']).reset_index()
df_cat_drop = df_cat.melt(id_vars=["count"]).drop_duplicates()

# summarise
df_cat_drop = pd.DataFrame(df_cat_drop.groupby(["variable"])["count"].sum()).sort_values(by=['count']).reset_index()

# keep columns with factors with 1 < x < 5
df_cat_drop=df_cat_drop[(df_cat_drop['count'] < 2) | (df_cat_drop['count'] > 4)]
df_cat_drop["drop"]=1
df_cat_drop=df_cat_drop.drop(columns=("count"))

# Merge
df_cat_keep2 = pd.merge(df_cat_keep,df_cat_drop,on=["variable"], how='left')
df_cat_keep2=df_cat_keep2[(df_cat_keep2['drop'] != 1)].sort_values(["variable","value"]).reset_index(drop=True)
df_cat_keep2=df_cat_keep2.drop(columns=("drop"))
df_cat_keep2

df_cat_keep2["value"].value_counts(dropna=False)

# rename or categorize variables
df_cat_keep2=df_cat_keep2.replace({"value": dict_value})
df_cat_keep2["category"]=df_cat_keep2["variable"]
df_cat_keep2=df_cat_keep2.replace({"category": dict_category})
df_cat_keep2["value"].value_counts(dropna=False)
df_cat_keep2

# graph
df_graph = (ggplot(df_cat_keep2, aes(y='count', x='value'))
+ geom_col(stat="identity")
+ coord_flip()
+ facet_wrap("~variable",scales="free")
+ theme(
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        text=element_text(size=7),
        subplots_adjust={'wspace':0.75, 'hspace':0.75}
)
)

ggsave(plot = df_graph, filename = "graph_categorical.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

del(df_graph,df_cat,df_cat_keep,df_cat_keep2,dict,df_cat_drop)

###########################
# Graph numerical variables
###########################

df_num = df.select_dtypes(include=[np.number])
df_num=df_num.drop(columns=["zip_code","latitude","longitude"])
df_num["count"]=1
df_num = df_num.melt(id_vars=["count"])
df_num

df_graph = (ggplot(df_num, aes(x='variable', y="value"))
+ geom_boxplot()
+ facet_wrap("~variable",scales="free")
+ theme(
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        # text=element_text(size=7),
        subplots_adjust={'wspace':0.25}
)
)

ggsave(plot = df_graph, filename = "graph_numerical.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)


###########################
# Descriptives
###########################

df.head()
df.info()
df.describe()
df["state"].value_counts()
df["churn_label"].value_counts()
df["churn_reason"].value_counts()
df_cat_keep["variable"].value_counts()

df


