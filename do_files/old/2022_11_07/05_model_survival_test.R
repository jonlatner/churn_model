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
# Survival Analysis with Lifelines¶
# must be a customer for at least one month

kmf = KaplanMeierFitter()

T = df['tenure_months']
E = df['churn_label']

kmf.fit(T, event_observed=E)

plt.clf() # this clears the figure
ax = plt.subplot(111)
kmf.survival_function_.plot(ax=ax,figsize=(10,6))

ax.grid()
plt.title('Survival Function of cusomers')
plt.xlabel('Months')
plt.show()
plt.savefig(os.path.join(main_dir,graphs,'km_curve.pdf'),bbox_inches='tight')

df_test = pd.DataFrame(kmf.survival_function_)
df_test.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different internet services
df["internet_service"].value_counts(dropna=False)

internet = ['Fiber optic', 'DSL', 'No']

plt.clf() # this clears the figure
ax = plt.subplot(111)

df_internet = pd.DataFrame({'A' : []})

for i in internet:
        kmf.fit(T[df['internet_service']== i], event_observed=E[df['internet_service']== i], label=i)
df_test = pd.DataFrame(kmf.survival_function_)
df_internet = pd.concat([df_internet, df_test],axis=1)
kmf.plot_survival_function(ax=ax,figsize=(10,6))

ax.grid()
plt.title('Lifespan of customers with different internet services')
plt.show()
plt.savefig(os.path.join(main_dir,graphs,'km_curve_internet.pdf'),bbox_inches='tight')

df_internet.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different payment methods
df["payment_method"].value_counts(dropna=False)

payment = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

plt.clf() # this clears the figure
ax = plt.subplot(111)

df_payment = pd.DataFrame({'A' : []})

for i in payment:
        kmf.fit(T[df['payment_method']== i], event_observed=E[df['payment_method']== i], label=i)
df_test = pd.DataFrame(kmf.survival_function_)
df_payment = pd.concat([df_payment, df_test],axis=1)
kmf.plot_survival_function(ax=ax,figsize=(10,6))

ax.grid()
plt.title('Lifespan of customers with different payment methods')
plt.show()
plt.savefig(os.path.join(main_dir,graphs,'km_curve_payment.pdf'),bbox_inches='tight')

df_payment.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different contract types
df["contract"].value_counts(dropna=False)

contract = ['Month-to-month', 'One year', 'Two year']

plt.clf() # this clears the figure
ax = plt.subplot(111)

df_contract = pd.DataFrame({'A' : []})
for i in contract:
        kmf.fit(T[df['contract']== i], event_observed=E[df['contract']== i], label=i)
df_test = pd.DataFrame(kmf.survival_function_)
df_contract = pd.concat([df_contract, df_test],axis=1)
kmf.plot_survival_function(ax=ax,figsize=(10,6))

ax.grid()
plt.title('Lifespan of customers with different contract types')
plt.show()
plt.savefig(os.path.join(main_dir,graphs,'km_curve_contract.pdf'),bbox_inches='tight')

df_contract.iloc[[1,12,24,36,48,60,72]]

###########################
# Cumulative hazard

naf = NelsonAalenFitter()
naf.fit(T, event_observed=E)
kmf.fit(T, event_observed=E)

plt.clf() # this clears the figure
ax = plt.subplot(111)

naf.plot_cumulative_hazard(ax=ax, label='cumulative hazard',figsize=(10,6))
kmf.plot_survival_function(ax=ax, label='survival function',figsize=(10,6))

plt.title('Survival Function VS Cumulative Hazard Function')
plt.show();

###########################
# Data preparation for Cox Proportional Hazard Function¶
dfcomplete = df.copy()

df = dfcomplete[["churn_label", 'tenure_months', 'monthly_charges', 'total_charges',
                 'dependents', 'gender', 'paperless_billing', 'partner', 'phone_service','senior_citizen',
                 # 'contract_month_to_month', 
                 'contract_one_year', 'contract_two_year',
                 # 'device_protection_no', 'device_protection_no_internet_service',
                 'device_protection_yes', 
                 # 'internet_service_dsl',
                 'internet_service_fiber_optic', 
                 'internet_service_no',
                 # 'multiple_lines_no', 'multiple_lines_no_phone_service',
                 'multiple_lines_yes', 
                 # 'online_backup_no','online_backup_no_internet_service', 
                 'online_backup_yes',
                 # 'online_security_no', 'online_security_no_internet_service',
                 'online_security_yes', 
                 #  'payment_method','payment_electronic_check', 
                 'payment_credit_card',
                 'payment_mailed_check','payment_method_bank_transfer_automatic_',
                 # 'streaming_movies_no', 'streaming_movies_no_internet_service',
                 'streaming_movies_yes', 
                 # 'streaming_tv_no','streaming_tv_no_internet_service', 
                 'streaming_tv_yes',
                 # 'tech_support_no', 'tech_support_no_internet_service',
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
fig = plt.gcf()  # or by other means, like plt.subplots
figsize = fig.set_size_inches(10, 6, forward=True)
plt.savefig(os.path.join(main_dir,graphs,'cph_coef.pdf'),bbox_inches='tight')

# CPH Model Visualization of survival curve for specific variables
plt.clf() # this clears the figure
ax = plt.subplot(111)

mylabels = ['contract two year', 'contract one year', "baseline"]

cph.plot_partial_effects_on_outcome(
        covariates=['contract_one_year',"contract_two_year"],
        values=[[0,1],[1,0]]
)
ax.grid()
plt.legend(labels=mylabels)
plt.title('CPH survival curve with different contract types')
plt.show()
plt.savefig(os.path.join(main_dir,graphs,'cph_curve_contract.pdf'),bbox_inches='tight')


