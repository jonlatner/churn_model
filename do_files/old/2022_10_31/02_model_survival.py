###########################
# TOP COMMANDS
###########################
# https://towardsdatascience.com/how-to-not-predict-and-prevent-customer-churn-1097c0a1ef3b
# https://www.kaggle.com/code/alessandromarceddu/churn-survival-analysis
# https://anzhi-tian.medium.com/customer-churn-analysis-and-prediction-with-survival-kmeans-and-lightgbm-algorithm-264503283d89
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np
import janitor # clean column names

## Algorithms
from sklearn import preprocessing
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

# Replace empty strings with NaN in column 'Name' 
df = df.replace(["^\s*$"], np.nan, regex=True)

# clean column names
df = pd.DataFrame.from_dict(df).clean_names().remove_empty()

# drop variables with only 1 factor (country, state, count)
# drop geographic variables with too many factors (customerid, lat_long, zip_code, latitude, longitude, city)
# drop variable with lots of missing (churn_reason)
df = df.drop(["customerid","country","state","lat_long","churn_reason","count","zip_code","latitude","longitude","city"], axis=1)


###########################
# Preprocessing Data for Customer Churn

# drop missings
df = df.dropna()

# Encoding categorical variables
cat_features = df.select_dtypes(exclude=np.number)
le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)
df_cat.head()

# selecting numerical variables
num_features = df.select_dtypes(include=np.number)
num_features.head()

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
