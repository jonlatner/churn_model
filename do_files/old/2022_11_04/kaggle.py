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

telecom = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))
telecom.describe()

telecom = telecom.drop(["Churn Reason","Count","Country","State", 
                        "City", "Zip Code", "Lat Long", "Latitude", "Longitude"],axis=1)

telecom = telecom.rename(columns={"Churn Label": "Churn"})


# Converting some binary variables (Yes/No) to 0/1¶

# List of variables to map

varlist =  ['Phone Service', 'Paperless Billing', 'Churn', 'Partner', 'Dependents', "Senior Citizen"]

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
telecom[varlist] = telecom[varlist].apply(binary_map)

#The varaible was imported as a string we need to convert it to float
# telecom['TotalCharges'] = telecom['TotalCharges'].astype(float) 
telecom['Total Charges'] = pd.to_numeric(telecom['Total Charges'], errors='coerce')

# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom[['Contract', 'Payment Method', 'Gender', 'Internet Service']], drop_first=True)

# Adding the results to the master dataframe
telecom = pd.concat([telecom, dummy1], axis=1)

# Creating dummy variables for the remaining categorical variables and dropping the level with big names.

# Creating dummy variables for the variable 'MultipleLines'
ml = pd.get_dummies(telecom['Multiple Lines'], prefix='Multiple Lines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['Multiple Lines_No phone service'], 1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ml1], axis=1)

# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['Online Security'], prefix='Online Security')
os1 = os.drop(['Online Security_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,os1], axis=1)

# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom['Online Backup'], prefix='Online Backup')
ob1 = ob.drop(['Online Backup_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1], axis=1)

# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom['Device Protection'], prefix='Device Protection')
dp1 = dp.drop(['Device Protection_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1], axis=1)

# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom['Tech Support'], prefix='Tech Support')
ts1 = ts.drop(['Tech Support_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1], axis=1)

# Creating dummy variables for the variable 'Streaming TV'.
st =pd.get_dummies(telecom['Streaming TV'], prefix='Streaming TV')
st1 = st.drop(['Streaming TV_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,st1], axis=1)

# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom['Streaming Movies'], prefix='Streaming Movies')
sm1 = sm.drop(['Streaming Movies_No internet service'], 1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1], axis=1)

# We have created dummies for the below variables, so we can drop them
telecom = telecom.drop(['Contract','Payment Method','Gender','Multiple Lines','Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
       'Tech Support', 'Streaming TV', 'Streaming Movies'], 1)
       
# Adding up the missing values (column-wise)
telecom.isnull().sum()

telecom = telecom.dropna()
telecom = telecom.reset_index(drop=True)


# Step 4: Test-Train Split¶
from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = telecom.drop(['Churn','CustomerID'], axis=1)

# Putting response variable to y
y = telecom['Churn']


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[['Tenure Months','Monthly Charges','Total Charges']] = scaler.fit_transform(X_train[['Tenure Months','Monthly Charges','Total Charges']])

X_train.head()

### Checking the Churn Rate
churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# Dropping highly correlated dummy variables¶


X_test = X_test.drop(['Multiple Lines_No','Online Security_No','Online Backup_No','Device Protection_No','Tech Support_No',
                       'Streaming TV_No','Streaming Movies_No'], 1)
X_train = X_train.drop(['Multiple Lines_No','Online Security_No','Online Backup_No','Device Protection_No','Tech Support_No',
                       'Streaming TV_No','Streaming Movies_No'], 1)
                         
# Step 7: Model Building¶
# Running Your First Training Model¶

import statsmodels.api as sm
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
