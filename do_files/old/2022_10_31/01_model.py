###########################
# TOP COMMANDS
###########################
# https://medium.com/@lucapetriconi/churn-modeling-a-detailed-step-by-step-guide-in-python-1e96d51c7523
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np
import janitor # clean column names
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

## Visualization
from plotnine import *
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

## ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve

## Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

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

df = df.drop(["customerid","country","state","lat_long","churn_reason","count","zip_code","latitude","longitude","city"], axis=1)

###########################
# Model
###########################

train, test = train_test_split(df, 
                                test_size=0.25, 
                                random_state=123)
X = train.drop(columns='churn_label', axis=1)
y = train['churn_label']

## Selecting categorical and numeric features
numerical_ix = X.select_dtypes(include=np.number).columns
categorical_ix = X.select_dtypes(exclude=np.number).columns

## Create preprocessing pipelines for each datatype
numerical_transformer = Pipeline(
                                steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                                ])

categorical_transformer = Pipeline(
                                steps=[
                                ('encoder', OrdinalEncoder()),
                                ('scaler', StandardScaler())
                                ])

## Putting the preprocessing steps together
preprocessor = ColumnTransformer([
                                ('numerical', numerical_transformer, numerical_ix),
                                ('categorical', categorical_transformer, categorical_ix)],
                                remainder='passthrough'
                                )


## Creat list of classifiers we're going to try out
classifiers = [
                KNeighborsClassifier(),
                SVC(random_state=123),
                DecisionTreeClassifier(random_state=123),
                RandomForestClassifier(random_state=123),
                AdaBoostClassifier(random_state=123),
                GradientBoostingClassifier(random_state=123)
                ]

classifier_names = [
                'KNeighborsClassifier()',
                'SVC()',
                'DecisionTreeClassifier()',
                'RandomForestClassifier()',
                'AdaBoostClassifier()',
                'GradientBoostingClassifier()'
                ]

model_scores = []

## Looping through the classifiers
for classifier, name in zip(classifiers, classifier_names):
  pipe = Pipeline(steps=[
                          ('preprocessor', preprocessor),
                          ('selector', SelectKBest(k=len(X.columns))),
                          ('classifier', classifier)
                          ])
  score = cross_val_score(pipe, X, y, cv=10, scoring='roc_auc').mean() 
  model_scores.append(score)
  
model_performance = pd.DataFrame({
                  'Classifier':
                    classifier_names,  
                  'Cross-validated AUC':
                    model_scores
                }).sort_values('Cross-validated AUC', ascending = False, ignore_index=True)

print(model_performance)

## Hyperparameter Tuning
# Let’s get our final pipeline:

pipe.get_params().keys()

pipe = Pipeline(steps=[
                  ('preprocessor', preprocessor),
                  ('selector', SelectKBest(k=len(X.columns))),
                  ('classifier', GradientBoostingClassifier(random_state=123))
                ])

grid = {
  # "selector__k": k_range,
  "classifier__max_depth":[1,3,5],
  "classifier__learning_rate":[0.01,0.1,1],
  "classifier__n_estimators":[100,200,300,400]
}

gridsearch = GridSearchCV(estimator=pipe, param_grid=grid, n_jobs= 1, scoring='roc_auc')

gridsearch.fit(X, y)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

# 4. Creating predictions for unseen data

## Separate features and target for the test data
X_test = test.drop(columns='churn_label', axis=1)
y_test = test['churn_label']

## Refitting the training data with the best parameters
gridsearch.refit

## Creating the predictions
y_pred = gridsearch.predict(X_test)
y_score = gridsearch.predict_proba(X_test)[:, 1]

## Looking at the performance
print('AUCROC:', roc_auc_score(y_test, y_score), '\nAccuracy:', accuracy_score(y_test, y_pred))

# Plotting the ROC curve
plot_roc_curve(gridsearch, X_test, y_test)
plt.show()
