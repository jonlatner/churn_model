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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

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

# scaler = StandardScaler()
# X_train[['tenure_months','monthly_charges','total_charges']] = scaler.fit_transform(X_train[['tenure_months','monthly_charges','total_charges']])

# Now, let’s oversample the training dataset:
# oversampling because only 27% churn (DV)
# oversample = SMOTE(k_neighbors=5)
# X_smote, y_smote = oversample.fit_resample(X_train, y_train)
# X_train, y_train = X_smote, y_smote
# y_train.value_counts()

###########################
# Building the Customer Churn Prediction Model 1

        X_train_1 = X_train.copy()
        X_train_1_sm = sm.add_constant(X_train_1)
        model_logit = sm.GLM(y_train,X_train_1_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        ###########################
        # Prepare the confusion matrix
        
        ## Getting the predicted values on the train set
        y_train_pred = results.predict(X_train_1_sm)
        y_train_pred[:10]
        
        ## Creating a dataframe with the actual churn flag and the predicted probabilities¶
        y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
        y_train_pred_final.head()
        
        ## Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
        cf_matrix
        output=cf_matrix/np.sum(cf_matrix)
        output
        
        ###########################
        # Table the confusion matrix
        
        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
        ACC = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)
        
        # set table to table to populate with performance results
        df_results["model_1"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_1.tex"),index=False, float_format="{:0.3f}".format)
        
        ###########################
        # Plot confusion matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        
        
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.clf() # this clears the figure
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        fig.set_size_inches(2.5,2.5)
        plt.savefig(os.path.join(main_dir,graphs,'model_glm_1.pdf'),bbox_inches='tight')
        
        ###########################
        # Check for the VIF values of the feature variables. 
        
        # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
        vif_1 = pd.DataFrame()
        vif_1['Features'] = X_train_1.columns
        vif_1['VIF_1'] = [variance_inflation_factor(X_train_1.values, i) for i in range(X_train_1.shape[1])]
        vif_1['VIF_1'] = round(vif_1['VIF_1'], 2)
        vif_1 = vif_1.sort_values(by = "VIF_1", ascending = False)
        vif_1
        
        vif_1.to_latex(os.path.join(main_dir,tables,"table_vif_1.tex"),index=False)

###########################
# Building the Customer Churn Prediction Model 2

        X_train_2 = X_train_1.drop(['monthly_charges'],axis=1)
        X_train_2_sm = sm.add_constant(X_train_2)
        model_logit = sm.GLM(y_train,X_train_2_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        ###########################
        # Prepare the confusion matrix
        
        ## Getting the predicted values on the train set
        y_train_pred = results.predict(X_train_2_sm)
        y_train_pred[:10]
        
        ## Creating a dataframe with the actual churn flag and the predicted probabilities¶
        y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
        y_train_pred_final.head()
        
        ## Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
        cf_matrix
        output=cf_matrix/np.sum(cf_matrix)
        output
        
        ###########################
        # Table the confusion matrix
        
        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
        ACC = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)
        
        # set table to table to populate with performance results
        df_results["model_2"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_2.tex"),index=False, float_format="{:0.3f}".format)
        
        ###########################
        # Plot confusion matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        
        
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.clf() # this clears the figure
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        fig.set_size_inches(2.5,2.5)
        plt.savefig(os.path.join(main_dir,graphs,'model_glm_2.pdf'),bbox_inches='tight')
        
        ###########################
        # Check for the VIF values of the feature variables. 
        
        # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
        vif_2 = pd.DataFrame()
        vif_2['Features'] = X_train_2.columns
        vif_2['VIF_2'] = [variance_inflation_factor(X_train_2.values, i) for i in range(X_train_2.shape[1])]
        vif_2['VIF_2'] = round(vif_2['VIF_2'], 2)
        vif_2 = vif_2.sort_values(by = "VIF_2", ascending = False)
        vif_2
        
        # Merge
        vif = pd.merge(vif_1, vif_2, how='outer', on='Features')
        vif.to_latex(os.path.join(main_dir,tables,"table_vif_2.tex"),index=False)

###########################
# Building the Customer Churn Prediction Model 3

        X_train_3 = X_train_2.drop(['total_charges'],axis=1)
        X_train_3_sm = sm.add_constant(X_train_3)
        model_logit = sm.GLM(y_train,X_train_3_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        ###########################
        # Prepare the confusion matrix
        
        ## Getting the predicted values on the train set
        y_train_pred = results.predict(X_train_3_sm)
        y_train_pred[:10]
        
        ## Creating a dataframe with the actual churn flag and the predicted probabilities¶
        y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
        y_train_pred_final.head()
        
        ## Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
        cf_matrix
        output=cf_matrix/np.sum(cf_matrix)
        output
        
        ###########################
        # Table the confusion matrix
        
        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
        ACC = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)
        
        # set table to table to populate with performance results
        df_results["model_3"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_3.tex"),index=False, float_format="{:0.3f}".format)
        
        ###########################
        # Plot confusion matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        
        
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.clf() # this clears the figure
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        fig.set_size_inches(2.5,2.5)
        plt.savefig(os.path.join(main_dir,graphs,'model_glm_3.pdf'),bbox_inches='tight')
        
        ###########################
        # Check for the VIF values of the feature variables. 
        
        # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
        vif_3 = pd.DataFrame()
        vif_3['Features'] = X_train_3.columns
        vif_3['VIF_3'] = [variance_inflation_factor(X_train_3.values, i) for i in range(X_train_3.shape[1])]
        vif_3['VIF_3'] = round(vif_3['VIF_3'], 2)
        vif_3 = vif_3.sort_values(by = "VIF_3", ascending = False)
        vif_3
        
        # Merge
        vif = pd.merge(vif, vif_3, how='outer', on='Features')
        vif.to_latex(os.path.join(main_dir,tables,"table_vif_3.tex"),index=False)
        vif
        
###########################
# Building the Customer Churn Prediction Model 4

        X_train_4 = X_train_3.drop(['phone_service'],axis=1)
        X_train_4_sm = sm.add_constant(X_train_4)
        model_logit = sm.GLM(y_train,X_train_4_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        ###########################
        # Prepare the confusion matrix
        
        ## Getting the predicted values on the train set
        y_train_pred = results.predict(X_train_4_sm)
        y_train_pred[:10]
        
        ## Creating a dataframe with the actual churn flag and the predicted probabilities¶
        y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
        y_train_pred_final.head()
        
        ## Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
        cf_matrix
        output=cf_matrix/np.sum(cf_matrix)
        output
        
        ###########################
        # Table the confusion matrix
        
        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
        ACC = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)
        
        # set table to table to populate with performance results
        df_results["model_4"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_4.tex"),index=False, float_format="{:0.3f}".format)
        
        ###########################
        # Plot confusion matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        
        
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.clf() # this clears the figure
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        fig.set_size_inches(2.5,2.5)
        plt.savefig(os.path.join(main_dir,graphs,'model_glm_4.pdf'),bbox_inches='tight')
        
        ###########################
        # Check for the VIF values of the feature variables. 
        
        # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
        vif_4 = pd.DataFrame()
        vif_4['Features'] = X_train_4.columns
        vif_4['VIF_4'] = [variance_inflation_factor(X_train_4.values, i) for i in range(X_train_4.shape[1])]
        vif_4['VIF_4'] = round(vif_4['VIF_4'], 2)
        vif_4 = vif_4.sort_values(by = "VIF_4", ascending = False)
        vif_4
        
        # Merge
        vif = pd.merge(vif, vif_4, how='outer', on='Features')
        vif.to_latex(os.path.join(main_dir,tables,"table_vif_4.tex"),index=False)
        vif
        
###########################
# Building the Customer Churn Prediction Model 5 with oversampling (i.e. model 1a)

        # Now, let’s oversample the training dataset:
        # oversampling because only 27% churn (DV)
        oversample = SMOTE(k_neighbors=5)
        X_smote, y_smote = oversample.fit_resample(X_train, y_train)
        X_train, y_train = X_smote, y_smote
        y_train.value_counts()
        
        X_train_5 = X_train.copy()
        X_train_5_sm = sm.add_constant(X_train_5)
        model_logit = sm.GLM(y_train,X_train_5_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        ###########################
        # Prepare the confusion matrix
        
        ## Getting the predicted values on the train set
        y_train_pred = results.predict(X_train_5_sm)
        y_train_pred[:10]
        
        ## Creating a dataframe with the actual churn flag and the predicted probabilities¶
        y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
        y_train_pred_final['CustID'] = y_train.index
        y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
        y_train_pred_final.head()
        
        ## Calculate confusion matrix
        cf_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
        cf_matrix
        output=cf_matrix/np.sum(cf_matrix)
        output
        
        ###########################
        # Table the confusion matrix
        
        ## Extract summary stats
        TN=output[0,0]
        FP=output[0,1]
        FN=output[1,0]
        TP=output[1,1]
        ACC = metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)
        
        # set table to table to populate with performance results
        df_results["model_5"] = (TN,FP,FN,TP,ACC)
        df_results
        df_results.to_latex(os.path.join(main_dir,tables,"table_cm_5.tex"),index=False, float_format="{:0.3f}".format)
        
        ###########################
        # Plot confusion matrix
        # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        
        
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = ["{0:0.0f}".format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        plt.clf() # this clears the figure
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        fig.set_size_inches(2.5,2.5)
        plt.savefig(os.path.join(main_dir,graphs,'model_glm_5.pdf'),bbox_inches='tight')
        
        ###########################
        # Check for the VIF values of the feature variables. 
        
        # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
        vif_5 = pd.DataFrame()
        vif_5['Features'] = X_train_5.columns
        vif_5['VIF_5'] = [variance_inflation_factor(X_train_5.values, i) for i in range(X_train_5.shape[1])]
        vif_5['VIF_5'] = round(vif_5['VIF_5'], 2)
        vif_5 = vif_5.sort_values(by = "VIF_5", ascending = False)

        # Merge
        vif = pd.merge(vif, vif_5, how='outer', on='Features')
        vif.to_latex(os.path.join(main_dir,tables,"table_vif_5.tex"),index=False)
        vif
