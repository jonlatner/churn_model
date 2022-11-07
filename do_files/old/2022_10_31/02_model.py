###########################
# TOP COMMANDS
###########################
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np
import janitor # clean column names

from sklearn.cluster import KMeans

## Algorithms
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

###########################
# clean
###########################

# clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

# Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)

df["churn_label"].value_counts(dropna=False)


###########################
# Preprocessing Data for Customer Churn
# Preprocessing Data for Customer Churn

# Determine number of factors for each categorical variable
cat_factors = df.select_dtypes(exclude=[np.number])
cat_factors["count"]=1
cat_factors = cat_factors.melt(id_vars=["count"]).drop_duplicates()
cat_factors=pd.DataFrame(cat_factors.groupby(["variable"])["count"].sum()).sort_values(by=['count']).reset_index()
cat_factors

# Identify and drop categorical variables with only 1 factor
drop_cat_vars = cat_factors[cat_factors["count"] == 1] 
drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
drop_cat_vars=list(drop_cat_vars.columns)
df = df.drop(columns=drop_cat_vars)

# Identify and drop categorical variables with more than 4 factors
drop_cat_vars = cat_factors[cat_factors["count"] > 4] 
drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
drop_cat_vars=list(drop_cat_vars.columns)
df = df.drop(columns=drop_cat_vars)

df.columns

# drop variables with only 1 factor (country, state, count)
# drop geographic variables with too many factors (customerid, lat_long, zip_code, latitude, longitude, city)
# drop variable with lots of missing (churn_reason)
df = df.drop(["customerid","country","state","lat_long","churn_reason","count","zip_code","latitude","longitude","city"], axis=1)


# selecting numerical variables
num_features = df.select_dtypes(include=np.number)
num_features.head()

###########################
# drop missings
df = df.dropna()


# Combine
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)

finaldf.describe()

# Independent and dependent variables
X = finaldf.drop(['churn_label'],axis=1)
y = finaldf['churn_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Now, let’s oversample the training dataset:
# oversampling because only 27% churn (DV)
oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

y_train.value_counts()


###########################
# Building the Customer Churn Prediction Model
rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)


###########################
# Customer Churn Prediction Model Evaluation

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))
