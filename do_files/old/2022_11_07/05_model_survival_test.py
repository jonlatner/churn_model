###########################
# TOP COMMANDS
###########################
# https://towardsdatascience.com/how-to-not-predict-and-prevent-customer-churn-1097c0a1ef3b
# https://www.kaggle.com/code/alessandromarceddu/churn-survival-analysis
# https://anzhi-tian.medium.com/customer-churn-analysis-and-prediction-with-survival-kmeans-and-lightgbm-algorithm-264503283d89

# https://stats.stackexchange.com/questions/533353/why-are-the-survival-curves-different-for-the-kaplan-meier-method-and-cox-regres

# create empty session
globals().clear()
globals().clear()

# load libraries
## Basics 
import os
import numpy as np 
import pandas as pd
import janitor # clean column names
import seaborn as sns

import matplotlib.pyplot as plt
import torch
import torchtuples as tt
from sklearn import preprocessing # one hot encoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.calibration import calibration_curve 

#lifelines
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines import NelsonAalenFitter
from lifelines import WeibullFitter
from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration

#pycox
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

#decision tree + random forest + fastai
import fastbook
fastbook.setup_book()
from fastbook import *
        from fastai.tabular.all import *
        from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import *
        from IPython.display import Image, display_svg, SVG

# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

###########################
# LOAD DATA

df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))

###########################
# Clean data

## clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

## Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)

df["internet_service"].value_counts(dropna=False)
df["customerid"].value_counts(dropna=False)
df["contract"].value_counts(dropna=False)

###########################
# Categorical variables

df_cat = df.select_dtypes(exclude=[np.number])

# Determine number of factors for each categorical variable
df_cat["count"]=1 
df_cat_factors = df_cat.melt(id_vars=["count"]).drop_duplicates()
df_cat_factors=pd.DataFrame(df_cat_factors.groupby(["variable"])["count"].sum()).sort_values(by=['count']).reset_index()
df_cat = df_cat.drop(columns="count")

# Identify and drop categorical variables with only 1 factor
drop_cat_vars = df_cat_factors[df_cat_factors["count"] == 1] 
drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
drop_cat_vars=list(drop_cat_vars.columns)
drop_cat_vars
df_cat = df_cat.drop(columns=drop_cat_vars)

# Identify and drop categorical variables with more than 4 factors
drop_cat_vars = df_cat_factors[df_cat_factors["count"] > 4] 
drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
drop_cat_vars=list(drop_cat_vars.columns)
drop_cat_vars
df_cat = df_cat.drop(columns=drop_cat_vars)

# Encode dummy variables with 2 factors
df_dummy_vars = df_cat_factors[df_cat_factors["count"] == 2]
df_dummy_vars = df_dummy_vars.pivot(index = "count", columns = "variable", values = "count")
df_dummy_vars=list(df_dummy_vars.columns)
df_dummy_vars = df_cat[(df_dummy_vars)]

## graph column names for later
dummy_vars_columns = df_dummy_vars.columns

## encodes alphabetically (no=0, yes=1; female=0,male=1)
df_dummy_vars.head()
le = preprocessing.LabelEncoder()
df_dummy_vars = df_dummy_vars.apply(le.fit_transform)
df_dummy_vars.head()

## replace unencoded dummy variables with new dummy variables
df_cat = df_cat.drop(columns=dummy_vars_columns)
df_cat = pd.merge(df_cat, df_dummy_vars, left_index=True, right_index=True)
df_cat.head()

# Create dataframe of dummy variables with 3 factors
df_dummy_vars = df_cat_factors[(df_cat_factors["count"] == 3) | (df_cat_factors["count"] == 4)]
df_dummy_vars = df_dummy_vars.pivot(index = "count", columns = "variable", values = "count")
df_dummy_vars=list(df_dummy_vars.columns)
df_dummy_vars = df_cat[(df_dummy_vars)]

## graph column names for later
dummy_vars_columns = df_dummy_vars.columns

## look at factor values/labels
df_test = df_dummy_vars.copy()
df_test["count"]=1 
df_test = df_test.melt(id_vars=["count"]).drop_duplicates()
df_test

## create dummy variables and clean column names
df_dummy_vars = pd.get_dummies(df_dummy_vars)
df_dummy_vars = pd.DataFrame.from_dict(df_dummy_vars).clean_names().remove_empty()

## select variables
df_dummy_vars.columns
df_dummy_vars = df_dummy_vars[[
        'contract_month_to_month', 'contract_one_year', 'contract_two_year',
        'device_protection_no', 'device_protection_no_internet_service',
        'device_protection_yes', 'internet_service_dsl',
        'internet_service_fiber_optic', 'internet_service_no',
        'multiple_lines_no', 'multiple_lines_no_phone_service',
        'multiple_lines_yes', 'online_backup_no',
        'online_backup_no_internet_service', 'online_backup_yes',
        'online_security_no', 'online_security_no_internet_service',
        'online_security_yes', 'payment_method_bank_transfer_automatic_',
        'payment_method_credit_card_automatic_',
        'payment_method_electronic_check', 'payment_method_mailed_check',
        'streaming_movies_no', 'streaming_movies_no_internet_service',
        'streaming_movies_yes', 'streaming_tv_no',
        'streaming_tv_no_internet_service', 'streaming_tv_yes',
        'tech_support_no', 'tech_support_no_internet_service',
        'tech_support_yes'
]]

## rename vars
df_dummy_vars = df_dummy_vars.rename(columns={
        "payment_method_mailed_check": "payment_mailed_check",
        "payment_method_credit_card_automatic_": "payment_credit_card",
        "payment_method_electronic_check": "payment_electronic_check"
})

## Drop and replace unencoded dummy variables with new dummy variables
# df_cat = df_cat.drop(columns=list(dummy_vars_columns))
df_cat = pd.merge(df_cat, df_dummy_vars, left_index=True, right_index=True)
# df_cat.head()

###########################
# Numerical variables

## selecting numerical variables
df_num = df.select_dtypes(include=np.number)
df_num = df_num.drop(columns=["zip_code","latitude","longitude"])

###########################
# Combine
finaldf = pd.merge(df_num, df_cat, left_index=True, right_index=True)

df_customer = df["customerid"]
finaldf = pd.merge(finaldf, df_customer, left_index=True, right_index=True)

finaldf.columns.sort_values()
finaldf = finaldf.dropna()

del(df,df_cat,df_num,df_test,df_dummy_vars,dummy_vars_columns,df_cat_factors)

df=finaldf.copy()
del(finaldf)

###########################
# Data preparation for Cox Proportional Hazard Function¶
dfcomplete = df.copy()

df = dfcomplete[[
                "churn_label", 'tenure_months', 
                'monthly_charges', 'total_charges',
                'dependents', 'gender', 'paperless_billing', 'partner', 'phone_service','senior_citizen',
                'contract_one_year', 'contract_two_year',
                'device_protection_yes',
                'internet_service_fiber_optic','internet_service_no',
                'multiple_lines_yes',
                'online_backup_yes',
                'online_security_yes',
                'payment_credit_card', 'payment_mailed_check','payment_method_bank_transfer_automatic_',
                'streaming_movies_yes',
                'streaming_tv_yes',
                'tech_support_yes'
                 ]]

df = dfcomplete[[
                "churn_label", 'tenure_months', 
                'monthly_charges','total_charges',
                'dependents', 'gender', 'partner','senior_citizen',
                'phone_service',
                'contract_one_year', 'contract_two_year',
                'device_protection_yes',
                'internet_service_fiber_optic','internet_service_no',
                'multiple_lines_yes',
                'online_backup_yes',
                'online_security_yes',
                'paperless_billing',
                'payment_credit_card', 'payment_mailed_check','payment_method_bank_transfer_automatic_',
                'streaming_movies_yes',
                'streaming_tv_yes',
                'tech_support_yes'
                 ]]

cph = CoxPHFitter()

cph.fit(df, duration_col='tenure_months', event_col='churn_label' )

# Print model summary
cph.print_summary(model = 'base model', decimals = 3, columns = ['coef', 'exp(coef)', 'p']) 

# CPH Model Visualization of all coefficients
plt.clf() # this clears the figure
ax = plt.subplot(111)
cph.plot(ax=ax)
ax.grid()
plt.show()
