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
from sklearn import preprocessing

## Visualization
from plotnine import *
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
        
# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

# https://medium.com/@lucapetriconi/churn-modeling-a-detailed-step-by-step-guide-in-python-1e96d51c7523
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/
# https://www.r-bloggers.com/2021/06/plotnine-make-great-looking-correlation-plots-in-python/

###########################
# LOAD DATA
###########################

df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))

# clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

df["churn_label"].value_counts(dropna=False)

###########################
# Missings
###########################

# Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)


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
df_variables

df_variables=df_variables[["number","variable","type","count"]]

# create table
df_variables.to_latex(os.path.join(main_dir,tables,"table_variables.tex"),index=False)
del(df_num,df_cat,df_variables)

###########################
# Recode bivariate variables
###########################

df["gender"].value_counts(dropna=False)
df["dependents"].value_counts(dropna=False)
df["senior_citizen"].value_counts(dropna=False)
df["partner"].value_counts(dropna=False)

df['female'] = np.where(df['gender'] == "Female", 1, 0)
df['children'] = np.where(df['dependents'] == "Yes", 1, 0)
df['senior'] = np.where(df['senior_citizen'] == "Yes", 1, 0)
df['hh_partner'] = np.where(df['partner'] == "Yes", 1, 0)


###########################
# Graph churn
###########################

df_graph = (ggplot(df, aes('churn_label', fill='churn_label'))
 + geom_bar()
 + geom_text(
     aes(label=after_stat('count')),
     stat='count',
     nudge_x=-0.14,
     nudge_y=0.125,
     va='bottom'
 )
 + geom_text(
     aes(label=after_stat('prop*100'), group=1),
     stat='count',
     nudge_x=0.14,
     nudge_y=0.125,
     va='bottom',
     format_string='({:.1f}%)'
 )
+ theme(
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        legend_position="none"
)
)

ggsave(plot = df_graph, filename = "graph_churn.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Graph churn by demographic variables
###########################

df_demographics = df[["churn_label",
                        "senior_citizen",
                        "gender",
                        "dependents",
                        "partner"]]
df_demographics["count"]=1
# rotate
df_demographics = df_demographics.melt(id_vars=["count","churn_label"])

df_demographics = pd.DataFrame(
                        (df_demographics
                        .groupby(["variable","churn_label","value"])["count"]
                        .sum())
                        .reset_index()
                        )
df_demographics['sum'] = df_demographics.groupby(["variable","churn_label"])['count'].transform(np.sum)
df_demographics["pct"] = df_demographics["count"]/df_demographics["sum"]

df_demographics

df_graph = (ggplot(df_demographics, aes(x="value", y = "pct", fill='churn_label', label = "pct"))
+ geom_bar(stat="identity",position=position_dodge(width=.9))
+ geom_text(va="bottom", format_string='{:.2f}',position=position_dodge(width=.9))
+ facet_wrap("~variable",nrow=1,scales="free_x")
+ scale_y_continuous(limits=(0,1))
+ theme(
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        legend_position="bottom",
        legend_title=element_blank()
        # subplots_adjust={'wspace':0.25, 'hspace':0.25}
)
)

ggsave(plot = df_graph, filename = "graph_churn_demographics.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Graph churn by services
###########################

df_services = df[["churn_label",
                "phone_service",
                "streaming_movies",
                "online_security",
                "multiple_lines",
                "internet_service",
                "device_protection",
                "streaming_tv",
                "streaming_tv"]]
df_services["count"]=1

df_services = df_services.rename(columns={
        'phone_service': 'phone \n service', 
        'streaming_movies': 'streaming \n movies',
        'online_security': 'online \n security',
        'multiple_lines': 'multiple \n lines',
        'internet_service': 'internet \n service',
        'device_protection': 'device \n protection',
        'streaming_tv': 'streaming \n tv',
        'streaming_tv': 'streaming \n tv',
})


# rotate
df_services = df_services.melt(id_vars=["count","churn_label"])

df_services = pd.DataFrame(
                        (df_services
                        .groupby(["variable","churn_label","value"])["count"]
                        .sum())
                        .reset_index()
                        )
df_services['sum'] = df_services.groupby(["variable","churn_label"])['count'].transform(np.sum)
df_services["pct"] = df_services["count"]/df_services["sum"]

df_services

df_graph = (ggplot(df_services, aes(x="value", y = "pct", fill='churn_label', label = "pct"))
+ geom_bar(stat="identity",position=position_dodge(width=.9))
# + geom_text(va="bottom", format_string='{:.2f}',position=position_dodge(width=.9))
+ facet_wrap("~variable",nrow=1,scales="free_x")
+ scale_y_continuous(limits=(0,1))
+ theme(
        axis_text_x = element_text(rotation=25, hjust=1),
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        # legend_position="bottom",
        legend_title=element_blank()
        # subplots_adjust={'wspace':0.25, 'hspace':0.25}
)
)

ggsave(plot = df_graph, filename = "graph_churn_services.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Graph churn by contract type
###########################

df_contract_type = df[["churn_label",
                  "contract",
                  "payment_method"
]]
df_contract_type["count"]=1

# rotate
df_contract_type = df_contract_type.melt(id_vars=["count","churn_label"])
df_contract_type

df_contract_type = pd.DataFrame(
        (df_contract_type
         .groupby(["variable","churn_label","value"])["count"]
         .sum())
        .reset_index()
)

df_contract_type['sum'] = df_contract_type.groupby(["variable","churn_label"])['count'].transform(np.sum)
df_contract_type["pct"] = df_contract_type["count"]/df_contract_type["sum"]

df_contract_type

df_graph = (ggplot(df_contract_type, aes(x="value", y = "pct", fill='churn_label', label = "pct"))
            + geom_bar(stat="identity",position=position_dodge(width=.9))
            + geom_text(va="bottom", format_string='{:.2f}',position=position_dodge(width=.9))
            + facet_wrap("~variable",nrow=1,scales="free_x")
            + scale_y_continuous(limits=(0,1))
            + theme(
                    axis_text_x = element_text(rotation=25, hjust=1),
                    axis_title_x = element_blank(),
                    axis_title_y = element_blank(),
                    # legend_position="bottom",
                    legend_title=element_blank()
                    # subplots_adjust={'wspace':0.25, 'hspace':0.25}
            )
)

ggsave(plot = df_graph, filename = "graph_churn_contract_type.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Graph churn by contract charges
###########################

df_contract_charges = df[["churn_label",
                  "tenure_months",
                  "monthly_charges",
                  "total_charges"
]]

# rotate
df_contract_charges = df_contract_charges.melt(id_vars=["churn_label"])
df_contract_charges


df_graph = (ggplot(df_contract_charges, aes(x="variable", y = "value", fill='churn_label'))
            + geom_boxplot(position=position_dodge(width=.9))
            + facet_wrap("~variable",nrow=1,scales="free")
            + theme(
                    axis_title_x = element_blank(),
                    axis_title_y = element_blank(),
                    # legend_position="bottom",
                    legend_title=element_blank(),
                    subplots_adjust={'wspace':0.25}
            )
)

ggsave(plot = df_graph, filename = "graph_churn_contract_charges.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Descriptives
###########################

cat_features = df.select_dtypes(exclude=[np.number])
cat_features.columns
cat_features

cat_features = df.drop(['customerid','total_charges','monthly_charges','senior_citizen','tenure_months'],axis=1)

