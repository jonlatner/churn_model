###########################
# TOP COMMANDS
###########################
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/
# create empty session
globals().clear()

# load libraries
import os
import pandas as pd
import numpy as np
import janitor # clean column names
from sklearn import preprocessing # one hot encoder
import matplotlib.pyplot as plt
import seaborn as sns

# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"


###########################
# LOAD DATA
###########################

df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))
df.describe()

df.columns

###########################
# clean
###########################

# clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

# Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)

# extract customer id
df_id = df[["customerid"]]

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

        ## rename vars
        df_dummy_vars = df_dummy_vars.rename(columns={
                "payment_method_bank_transfer_automatic_": "payment_bank_transfer_auto",
                "payment_method_mailed_check": "payment_mailed_check",
                "payment_method_credit_card_automatic_": "payment_credit_card",
                "payment_method_electronic_check": "payment_electronic_check"
                })

        ## Drop and replace unencoded dummy variables with new dummy variables
        # df_cat = df_cat.drop(columns=list(dummy_vars_columns))
        df_cat = pd.merge(df_cat, df_dummy_vars, left_index=True, right_index=True)
        df_cat.columns.sort_values()

        ## select variables
        df_cat_corr = df_cat[[
        'churn_label', 
        'dependents', 'gender', 'partner','senior_citizen', 
        'phone_service',  'internet_service_no',
        'paperless_billing', 
        'contract_month_to_month','contract_one_year', 'contract_two_year',
        'payment_bank_transfer_auto', 'payment_mailed_check','payment_electronic_check', "payment_credit_card",
        'device_protection_yes',
        'internet_service_fiber_optic',
        'multiple_lines_yes',
        'online_backup_yes',
        'online_security_yes',
        'streaming_movies_yes', 
        'streaming_tv_yes', 
        'tech_support_yes'
       ]]

        ## Let's see the correlation matrix 
        plt.figure(figsize = (10,10))        # Size of the figure
        sns.heatmap(df_cat_corr.corr(),
                        annot = True, fmt='.1f',
                        vmin=-1, vmax=1)
        plt.tight_layout()
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        figsize = fig.get_size_inches()
        fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5
        
        plt.savefig(os.path.join(main_dir,graphs,'multivaritate_corr.pdf'),bbox_inches='tight')
        
        df_corr = df_cat_corr.corr()

del(df_cat_factors,drop_cat_vars,df_dummy_vars,dummy_vars_columns,le,df_test,df_cat_corr,df_corr)

###########################
# Numerical variables

        # selecting numerical variables
        df_num = df.select_dtypes(include=np.number)
        df_num.head()

        # identify and drop numerical variables with 0 variance
        df_num_var_drop = df_num.agg(["var"]).reset_index()
        df_num_var_drop = df_num_var_drop.melt(id_vars=["index"]).drop_duplicates()
        df_num_var_drop = df_num_var_drop.drop(columns="index")
        df_num_var_drop = df_num_var_drop[df_num_var_drop["value"] == 0] 
        df_num_var_drop = df_num_var_drop.pivot(index = "value", columns = "variable", values = "value")
        df_num_var_drop=list(df_num_var_drop.columns)
        df_num = df_num.drop(columns=df_num_var_drop)

        df_num = df_num.drop(columns=["zip_code","latitude","longitude"])

        # Checking for outliers in the continuous variables
        df_num.describe(percentiles=[.25, .5, .75, .90, .95, .99])

###########################
# categorize tenure months into years
        tenure_years = pd.cut(df['tenure_months'],
                                bins=[0,12,24,36,48,60,99],
                                labels=[1,2,3,4,5,6]
                                )
        tenure_years.value_counts(dropna=False)
        
        df_dummy_years = pd.get_dummies(tenure_years, prefix = "tenure")
        
        
###########################
# Combine
        finaldf = pd.merge(df_num, df_cat, left_index=True, right_index=True)
        
        # add in customer id
        finaldf = pd.merge(finaldf, df_id, left_index=True, right_index=True)
        
        # add in years
        finaldf = pd.merge(finaldf, df_dummy_years, left_index=True, right_index=True)
        
        finaldf.columns.sort_values()

###########################
# Graph
## Let's see the correlation matrix 

        ## select variables
        df_cat_corr = finaldf[[
        "tenure_months","monthly_charges","total_charges",
        'churn_label', 
        'dependents', 'gender', 'partner','senior_citizen', 
        'phone_service',  'internet_service_no',
        'paperless_billing', 
        'contract_month_to_month','contract_one_year', 'contract_two_year',
        'payment_bank_transfer_auto', 'payment_mailed_check','payment_electronic_check', "payment_credit_card",
        'device_protection_yes',
        'internet_service_fiber_optic',
        'multiple_lines_yes',
        'online_backup_yes',
        'online_security_yes',
        'streaming_movies_yes', 
        'streaming_tv_yes', 
        'tech_support_yes'
       ]]

        corr = df_cat_corr.corr().iloc[1:,:-1].copy()

        # Generate annotation labels array (of the same size as the heatmap data)- filling cells you don't want to annotate with an empty string ''
        annot_labels = np.empty_like(corr, dtype=str)
        annot_mask = np.where(corr >= .7,round(corr,2),
                                np.where(corr <= -.7,round(corr,2),
                                " "))

        # Create a mask
        mask = np.triu(np.ones_like(df_cat_corr.corr(), dtype=bool))
        # adjust mask and df
        mask = mask[1:, :-1]

        plt.figure(figsize = (10,10))        # Size of the figure
        sns.heatmap(corr,
                        mask=mask,
                        annot=annot_mask,
                        fmt='',
                        cmap='Blues',
                        # annot = True,
                        # fmt='.2f',
                        vmin=-1, vmax=1)
        plt.tight_layout()
        plt.show()
        
        fig = plt.gcf()  # or by other means, like plt.subplots
        figsize = fig.get_size_inches()
        fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5
        plt.savefig(os.path.join(main_dir,graphs,'multivaritate_corr_finaldf.pdf'),bbox_inches='tight')

###########################
# drop missings

        # missing values?
        missings=pd.DataFrame(finaldf.isnull().sum()).reset_index()
        missings.columns =['variable', 'count']
        missings[missings["count"] > 0] 

        finaldf = finaldf.dropna()


###########################
# Save

finaldf.to_csv(os.path.join(main_dir,data_files,"churn_model.csv"))  

finaldf.groupby('contract')['tenure_months'].mean().reset_index()
