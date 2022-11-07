###########################
# TOP COMMANDS
###########################
# https://towardsdatascience.com/how-to-not-predict-and-prevent-customer-churn-1097c0a1ef3b
# https://www.kaggle.com/code/alessandromarceddu/churn-survival-analysis
# https://anzhi-tian.medium.com/customer-churn-analysis-and-prediction-with-survival-kmeans-and-lightgbm-algorithm-264503283d89

# https://stats.stackexchange.com/questions/533353/why-are-the-survival-curves-different-for-the-kaplan-meier-method-and-cox-regres

# create empty session
globals().clear()
globals().clear()

# load libraries
## Basics 
import os
import numpy as np 
import pandas as pd

## Graphs
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

#lifelines
import lifelines
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import CoxPHFitter

# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

###########################
# LOAD DATA

        df = pd.read_csv(os.path.join(main_dir,data_files,"churn_model.csv"), index_col=0)
        df.columns.sort_values()

###########################
# Survival Analysis with Lifelines¶
# must be a customer for at least one month

        kmf = KaplanMeierFitter()
        
        T = df['tenure_months']
        E = df['churn_label']
        
        kmf.fit(T, event_observed=E)
        
        plt.clf() # this clears the figure
        ax = plt.subplot(111)
        kmf.survival_function_.plot(ax=ax,figsize=(10,6))
        
        ax.grid()
        plt.title('Survival Function of cusomers')
        plt.xlabel('Months')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'km_curve.pdf'),bbox_inches='tight')
        
        df_test = pd.DataFrame(kmf.survival_function_)
        df_test.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with phone service
        df["phone_service"].value_counts(dropna=False)

        phone = [1, 0]

        plt.clf() # this clears the figure
        ax = plt.subplot(111)

        df_phone = pd.DataFrame({'A' : []})

for i in phone:
        kmf.fit(T[df['phone_service']== i], event_observed=E[df['phone_service']== i], label=i)
        df_test = pd.DataFrame(kmf.survival_function_)
        df_phone = pd.concat([df_phone, df_test],axis=1)
        kmf.plot_survival_function(ax=ax,figsize=(10,6))

        ax.grid()
        plt.title('Lifespan of customers with phone services')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'km_curve_phone.pdf'),bbox_inches='tight')

        df_phone.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different internet services
        df["internet_service"].value_counts(dropna=False)

        internet = ['Fiber optic', 'DSL', 'No']

        plt.clf() # this clears the figure
        ax = plt.subplot(111)

        df_internet = pd.DataFrame({'A' : []})

for i in internet:
        kmf.fit(T[df['internet_service']== i], event_observed=E[df['internet_service']== i], label=i)
        df_test = pd.DataFrame(kmf.survival_function_)
        df_internet = pd.concat([df_internet, df_test],axis=1)
        kmf.plot_survival_function(ax=ax,figsize=(10,6))

        ax.grid()
        plt.title('Lifespan of customers with different internet services')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'km_curve_internet.pdf'),bbox_inches='tight')

        df_internet.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different payment methods
        df["payment_method"].value_counts(dropna=False)

        payment = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

        plt.clf() # this clears the figure
        ax = plt.subplot(111)

        df_payment = pd.DataFrame({'A' : []})

for i in payment:
        kmf.fit(T[df['payment_method']== i], event_observed=E[df['payment_method']== i], label=i)
        df_test = pd.DataFrame(kmf.survival_function_)
        df_payment = pd.concat([df_payment, df_test],axis=1)
        kmf.plot_survival_function(ax=ax,figsize=(10,6))

        ax.grid()
        plt.title('Lifespan of customers with different payment methods')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'km_curve_payment.pdf'),bbox_inches='tight')

        df_payment.iloc[[1,12,24,36,48,60,72]]

## Lifespan of customers with different contract types
        df["contract"].value_counts(dropna=False)

        contract = ['Month-to-month', 'One year', 'Two year']

        plt.clf() # this clears the figure
        ax = plt.subplot(111)

        df_contract = pd.DataFrame({'A' : []})
for i in contract:
        kmf.fit(T[df['contract']== i], event_observed=E[df['contract']== i], label=i)
        df_test = pd.DataFrame(kmf.survival_function_)
        df_contract = pd.concat([df_contract, df_test],axis=1)
        kmf.plot_survival_function(ax=ax,figsize=(10,6))

        ax.grid()
        plt.title('Lifespan of customers with different contract types')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'km_curve_contract.pdf'),bbox_inches='tight')

        df_contract.iloc[[1,12,24,36,48,60,72]]

###########################
# Cumulative hazard

        naf = NelsonAalenFitter()
        naf.fit(T, event_observed=E)
        kmf.fit(T, event_observed=E)
        
        plt.clf() # this clears the figure
        ax = plt.subplot(111)
        
        naf.plot_cumulative_hazard(ax=ax, label='cumulative hazard',figsize=(10,6))
        kmf.plot_survival_function(ax=ax, label='survival function',figsize=(10,6))
        
        plt.title('Survival Function VS Cumulative Hazard Function')
        plt.show();

###########################
# Data preparation for Cox Proportional Hazard Function¶
dfcomplete = df.copy()

# drop reference categories and total_charges, monthly_charges, phone_service
dfcomplete = dfcomplete[[
        'churn_label', "tenure_months",
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
        'streaming_movies_yes',
        'streaming_tv_yes',
        'tech_support_yes',
       ]]

cph = CoxPHFitter()

cph.fit(dfcomplete, duration_col='tenure_months', event_col='churn_label' )

# Print model summary
cph.print_summary(model = 'base model', decimals = 3, columns = ['coef', 'exp(coef)', 'p']) 

# CPH Model Visualization of all coefficients
        plt.clf() # this clears the figure
        ax = plt.subplot(111)
        cph.plot(ax=ax)
        ax.grid()
        plt.show()
        fig = plt.gcf()  # or by other means, like plt.subplots
        figsize = fig.set_size_inches(10, 6, forward=True)
        plt.savefig(os.path.join(main_dir,graphs,'cph_coef.pdf'),bbox_inches='tight')

# CPH Model Visualization of survival curve for specific variables
        plt.clf() # this clears the figure
        ax = plt.subplot(111)

        mylabels = ['contract two year', 'contract one year', "baseline"]

        cph.plot_partial_effects_on_outcome(
                covariates=['contract_one_year',"contract_two_year"],
                values=[[0,1],[1,0]]
        )
        ax.grid()
        plt.legend(labels=mylabels)
        plt.title('CPH survival curve with different contract types')
        plt.show()
        plt.savefig(os.path.join(main_dir,graphs,'cph_curve_contract.pdf'),bbox_inches='tight')


