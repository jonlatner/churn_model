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
from plotnine import *
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

## Algorithms
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

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

###########################
# Prepare data

## Independent and dependent variables
X = df.drop(['churn_label',"customerid"],axis=1)

y = df['churn_label']

## Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# standardize continuous variables, so mean = 0, and var = 1
# scaler = StandardScaler()
# X_train[['tenure_months','monthly_charges','total_charges']] = scaler.fit_transform(X_train[['tenure_months','monthly_charges','total_charges']])

# Now, let’s oversample the training dataset:
# oversampling because only 27% churn (DV)
# oversample = SMOTE(k_neighbors=5)
# X_smote, y_smote = oversample.fit_resample(X_train, y_train)
# X_train, y_train = X_smote, y_smote
# y_train.value_counts()
# 
###########################
# Building the Customer Churn Prediction Model

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
        
        ###########################
        # Step 10: Finding Optimal Cutoff Point¶

        ## Let's create columns with different probability cutoffs 
        numbers = [float(x)/10 for x in range(10)]
        for i in numbers:
            y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)

        y_train_pred_final.head()

        ## Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
        cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

        num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        for i in num:
            cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
            total1=sum(sum(cm1))
            accuracy = (cm1[0,0]+cm1[1,1])/total1
            
            speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

        print(cutoff_df)
        cutoff_df = cutoff_df.melt(id_vars=["prob"])
        print(cutoff_df)

        ## Graph
        
        df_graph = (ggplot(cutoff_df, aes(x="prob", y = "value", color='variable', group="variable"))
        + geom_line(size=1)
        + scale_y_continuous(limits=(0,1.01), breaks=np.arange(0, 1.01, 0.1))
        + scale_x_continuous(limits=(0,1.01), breaks=np.arange(0, 1.01, 0.1))
        + theme(
                axis_title_x = element_blank(),
                axis_title_y = element_blank(),
                # legend_position="bottom",
                legend_title=element_blank(),
                legend_position=(.5, -.1)
        )
        )
        
        ggsave(plot = df_graph, filename = "graph_cutoff.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)
