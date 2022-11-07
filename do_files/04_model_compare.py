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
from plotnine import *

## Algorithms
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score, auc, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
df.columns.sort_values()

# drop reference categories
df = df[[
        'churn_label', 
        'monthly_charges',"total_charges",'tenure_months',
        'senior_citizen','dependents','gender','partner',
        # 'contract_month_to_month',
        'contract_one_year', 'contract_two_year', 
        'device_protection_yes',
        'internet_service_fiber_optic', 'internet_service_no',
        'multiple_lines_yes',
        'online_backup_yes',
        'online_security_yes',
        'paperless_billing',
        # 'payment_mailed_check',
        'payment_bank_transfer_auto','payment_credit_card', 'payment_electronic_check',
        'phone_service',
        'streaming_movies_yes',
        'streaming_tv_yes',
        'tech_support_yes',
       ]]

df["churn_label"].value_counts(dropna=False)

# set table to table to populate with performance results
data = ({'measure':["TN", "FP", "FN", "TP","ACC"]})
df_results = pd.DataFrame(data, columns=['measure'])
del(data)

###########################
# Prepare data

## Independent and dependent variables
X = df.drop(['churn_label', "total_charges", "monthly_charges", "phone_service"],axis=1)

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

###########################
# Plot logistic regression

        X_train_1_sm = sm.add_constant(X_train)
        model_logit = sm.GLM(y_train,X_train_1_sm, family = sm.families.Binomial())
        model_logit.fit().summary()
        results = model_logit.fit()
        
        # model_ols = sm.OLS(y_train,X_train_1_sm)
        # model_ols.fit().summary()
        # results = model_ols.fit()

        coef_df = pd.DataFrame({
                'varname': results.params.index,
                'coef': results.params.values,
                'lo': results.conf_int().values[:, 0],
                'hi': results.conf_int().values[:, 1],
                })

        coef_df = coef_df.query("varname != 'const'")
        coef_df['coef'] = np.exp(coef_df['coef'])
        coef_df['lo'] = np.exp(coef_df['lo'])
        coef_df['hi'] = np.exp(coef_df['hi'])
        coef_df


        df_graph = (ggplot(coef_df, aes(x='reorder(varname, coef)', y = "coef"))
                    + geom_point()
                    + geom_errorbar(aes(x='reorder(varname, coef)', ymin="lo",ymax="hi"))                    
                    + geom_hline(yintercept = 1) # add one horizonal line
                    + coord_flip()
                    + theme(
                            axis_title_x = element_blank(),
                            axis_title_y = element_blank(),
                            # legend_position="bottom",
                            legend_title=element_blank(),
                            subplots_adjust={'wspace':0.25}
                    )
                    )
        df_graph
        
        ggsave(plot = df_graph, filename = "graph_glm.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 6)
