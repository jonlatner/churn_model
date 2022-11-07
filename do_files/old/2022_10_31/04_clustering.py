###########################
# TOP COMMANDS
###########################
# https://www.kaggle.com/code/foxtrotoscar/eda-clustering-predicting-churn
# https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3
# https://anzhi-tian.medium.com/customer-churn-analysis-and-prediction-with-survival-kmeans-and-lightgbm-algorithm-264503283d89
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np

## Algorithms
import statsmodels.api as sm

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
# Load data

df = pd.read_csv(os.path.join(main_dir,data_files,"churn_model.csv"), index_col=0)
df.describe()

df["churn_label"].value_counts(dropna=False)

###########################
# Prepare data

# Independent and dependent variables
X = df.drop(['churn_label'],axis=1)
y = df['churn_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Now, let’s oversample the training dataset:
# oversampling because only 27% churn (DV)
oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

y_train.value_counts()


###########################
# Building the Customer Churn Prediction Model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

result = logit.fit()
result.summary()

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)

preds = result.predict(X_test)
print(accuracy_score(preds,y_test))


rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)


###########################
# Customer Churn Prediction Model Evaluation

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))
