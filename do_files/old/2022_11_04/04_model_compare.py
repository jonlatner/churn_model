###########################
# TOP COMMANDS
###########################
# https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/
# https://www.kaggle.com/code/gauravduttakiit/telecom-churn-case-study-logistic-regression/notebook
# https://towardsdatascience.com/predict-customer-churn-in-python-e8cd6d3aaa7
# create empty session
globals().clear()

# load libraries
## Basics 
import os
import pandas as pd
import numpy as np
from pystout import pystout

## Graphs
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

## Algorithms
# import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, auc, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
df.columns

df["churn_label"].value_counts(dropna=False)

# set table to table to populate with performance results
data = ({'measure':["TN", "FP", "FN", "TP","ACC"]})
df_results = pd.DataFrame(data, columns=['measure'])
del(data)

###########################
# Prepare data

## Independent and dependent variables
X = df.drop(['churn_label',"customerid"],axis=1)

y = df['churn_label']

## Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Now, let’s oversample the training dataset:
# oversampling because only 27% churn (DV)
oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote
y_train.value_counts()

###########################
# Fitting Logistic Regression to the Training set

        classifier = LogisticRegression(max_iter=10000)
        classifier.fit(X_train, y_train)
        
        ## Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        ##Table the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        output=cf_matrix/np.sum(cf_matrix)

        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
                
        ##Evaluate results
        ACC = accuracy_score(y_test, y_pred )
        
        ## set table to table to populate with performance results
        df_results["GLM"] = (TN,FP,FN,TP,ACC)
        df_results

###########################
# Fitting SVM (SVC class) to the Training set
        # classifier = SVC(kernel = 'linear', random_state = 0)
        # classifier.fit(X_train, y_train)
        # 
        # ## Predicting the Test set results
        # y_pred = classifier.predict(X_test)
        # 
        # ##Table the confusion matrix
        # cf_matrix = confusion_matrix(y_test, y_pred)
        # output=cf_matrix/np.sum(cf_matrix)
        # 
        # ## Extract summary stats
        # TN=output[0,0]
        # FP=output[0,1]
        # FN=output[1,0]
        # TP=output[1,1]
        #         
        # ##Evaluate results
        # ACC = accuracy_score(y_test, y_pred )
        # 
        # ## set table to table to populate with performance results
        # df_results["SVM"] = (TN,FP,FN,TP,ACC)
        # df_results
        # # df_results.to_latex(os.path.join(main_dir,tables,"table_cm_1.tex"),index=False, float_format="{:0.3f}".format)

###########################
# Fitting KNN to the Training set:

        classifier = KNeighborsClassifier(
                                n_neighbors = 22, 
                                metric = 'minkowski', p = 2)

        classifier.fit(X_train, y_train)

        ## Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        ##Table the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        output=cf_matrix/np.sum(cf_matrix)

        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
                
        ##Evaluate results
        ACC = accuracy_score(y_test, y_pred )
        
        ## set table to table to populate with performance results
        df_results["KNN"] = (TN,FP,FN,TP,ACC)
        df_results
        # df_results.to_latex(os.path.join(main_dir,tables,"table_cm_1.tex"),index=False, float_format="{:0.3f}".format)

###########################
# Fitting Naive Byes to the Training set:

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        ## Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        ##Table the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        output=cf_matrix/np.sum(cf_matrix)

        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
                
        ##Evaluate results
        ACC = accuracy_score(y_test, y_pred )
        
        ## set table to table to populate with performance results
        df_results["NB"] = (TN,FP,FN,TP,ACC)
        df_results
        # df_results.to_latex(os.path.join(main_dir,tables,"table_cm_1.tex"),index=False, float_format="{:0.3f}".format)

###########################
# Fitting Decision Tree to the Training set:

        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)

        ## Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        ##Table the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        output=cf_matrix/np.sum(cf_matrix)

        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
                
        ##Evaluate results
        ACC = accuracy_score(y_test, y_pred )
        
        ## set table to table to populate with performance results
        df_results["DT"] = (TN,FP,FN,TP,ACC)
        df_results
        # df_results.to_latex(os.path.join(main_dir,tables,"table_cm_1.tex"),index=False, float_format="{:0.3f}".format)

###########################
# Fitting Random Forest to the Training set:
    
        classifier = RandomForestClassifier(n_estimators = 72, 
                                        criterion = 'entropy', random_state = 0
                                        )

        classifier.fit(X_train, y_train)

        ## Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        ##Table the confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)
        output=cf_matrix/np.sum(cf_matrix)

        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
                
        ##Evaluate results
        ACC = accuracy_score(y_test, y_pred )
        
        ## set table to table to populate with performance results
        df_results["RF"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_compare.tex"),index=False, float_format="{:0.3f}".format)
