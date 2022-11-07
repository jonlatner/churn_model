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
import janitor # clean column names

## Graphs
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

## Algorithms
from sklearn import preprocessing # one hot encoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# beginning commands
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/churn_model/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

###########################
# LOAD DATA

        df = pd.read_excel(os.path.join(main_dir,data_files,"Customer Churn.xlsx"))

###########################
# Clean data

        ## clean column names
        df = pd.DataFrame.from_dict(df).clean_names().remove_empty()
        
        ## Replace empty strings with NaN in column 'Name' 
        df = df.replace(["^\s*$"], np.nan, regex=True)
        
        df["internet_service"].value_counts(dropna=False)
        df["customerid"].value_counts(dropna=False)
        df["contract"].value_counts(dropna=False)

        ## rename customerid
        df["customerid"] = np.arange(len(df))+1

###########################
# Categorical variables

        df_cat = df.select_dtypes(exclude=[np.number])

# Determine number of factors for each categorical variable
        df_cat["count"]=1 
        df_cat_factors = df_cat.melt(id_vars=["count"]).drop_duplicates()
        df_cat_factors=pd.DataFrame(df_cat_factors.groupby(["variable"])["count"].sum()).sort_values(by=['count']).reset_index()
        df_cat = df_cat.drop(columns="count")

# Identify and drop categorical variables with only 1 factor
        drop_cat_vars = df_cat_factors[df_cat_factors["count"] == 1] 
        drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
        drop_cat_vars=list(drop_cat_vars.columns)
        drop_cat_vars
        df_cat = df_cat.drop(columns=drop_cat_vars)

# Identify and drop categorical variables with more than 4 factors
        drop_cat_vars = df_cat_factors[df_cat_factors["count"] > 4] 
        drop_cat_vars = drop_cat_vars.pivot(index = "count", columns = "variable", values = "count")
        drop_cat_vars=list(drop_cat_vars.columns)
        drop_cat_vars
        df_cat = df_cat.drop(columns=drop_cat_vars)

# Encode dummy variables with 2 factors
        df_dummy_vars = df_cat_factors[df_cat_factors["count"] == 2]
        df_dummy_vars = df_dummy_vars.pivot(index = "count", columns = "variable", values = "count")
        df_dummy_vars=list(df_dummy_vars.columns)
        df_dummy_vars = df_cat[(df_dummy_vars)]

        ## graph column names for later
        dummy_vars_columns = df_dummy_vars.columns
        
        ## encodes alphabetically (no=0, yes=1; female=0,male=1)
        df_dummy_vars.head()
        le = preprocessing.LabelEncoder()
        df_dummy_vars = df_dummy_vars.apply(le.fit_transform)
        df_dummy_vars.head()
        
        ## replace unencoded dummy variables with new dummy variables
        df_cat = df_cat.drop(columns=dummy_vars_columns)
        df_cat = pd.merge(df_cat, df_dummy_vars, left_index=True, right_index=True)
        df_cat.head()

# Create dataframe of dummy variables with 3 factors
        df_dummy_vars = df_cat_factors[(df_cat_factors["count"] == 3) | (df_cat_factors["count"] == 4)]
        df_dummy_vars = df_dummy_vars.pivot(index = "count", columns = "variable", values = "count")
        df_dummy_vars=list(df_dummy_vars.columns)
        df_dummy_vars = df_cat[(df_dummy_vars)]
        
        ## graph column names for later
        dummy_vars_columns = df_dummy_vars.columns

        ## look at factor values/labels
        df_test = df_dummy_vars.copy()
        df_test["count"]=1 
        df_test = df_test.melt(id_vars=["count"]).drop_duplicates()
        df_test
        
        ## create dummy variables and clean column names
        df_dummy_vars = pd.get_dummies(df_dummy_vars)
        df_dummy_vars = pd.DataFrame.from_dict(df_dummy_vars).clean_names().remove_empty()

        ## select variables
        df_dummy_vars.columns
        df_dummy_vars = df_dummy_vars[[
                'contract_month_to_month', 'contract_one_year', 'contract_two_year',
                # 'device_protection_no', 'device_protection_no_internet_service',
                'device_protection_yes', 
                # 'internet_service_dsl',
                'internet_service_fiber_optic', 'internet_service_no',
                # 'multiple_lines_no', 'multiple_lines_no_phone_service',
                'multiple_lines_yes', 
                # 'online_backup_no', 'online_backup_no_internet_service', 
                'online_backup_yes',
                # 'online_security_no', 'online_security_no_internet_service',
                'online_security_yes', 
                'payment_method_bank_transfer_automatic_','payment_method_credit_card_automatic_',
                'payment_method_electronic_check', 'payment_method_mailed_check',
                # 'streaming_movies_no','streaming_movies_no_internet_service',
                'streaming_movies_yes', 
                # 'streaming_tv_no', 'streaming_tv_no_internet_service', 
                'streaming_tv_yes',
                # 'tech_support_no', 'tech_support_no_internet_service',
                'tech_support_yes'
                ]]
        
        ## rename vars
        df_dummy_vars = df_dummy_vars.rename(columns={
                "payment_method_bank_transfer_automatic_": "payment_bank_transfer_auto",
                "payment_method_mailed_check": "payment_mailed_check",
                "payment_method_credit_card_automatic_": "payment_credit_card",
                "payment_method_electronic_check": "payment_electronic_check"
                })

        ## Drop and replace unencoded dummy variables with new dummy variables
        df_cat = df_cat.drop(columns=list(dummy_vars_columns))
        df_cat = pd.merge(df_cat, df_dummy_vars, left_index=True, right_index=True)
        # df_cat.head()

###########################
# Numerical variables

        ## selecting numerical variables
        df_num = df.select_dtypes(include=np.number)
        df_num = df_num.drop(columns=["zip_code","latitude","longitude"])

###########################
# Combine
        finaldf = pd.merge(df_num, df_cat, left_index=True, right_index=True)

        finaldf.columns.sort_values()
        finaldf = finaldf.dropna()
        
        del(df,df_cat,df_num,df_test,df_dummy_vars,dummy_vars_columns,df_cat_factors)
        
        df=finaldf.copy()
        del(finaldf)
        
        df.columns.sort_values()

        df = df.drop(["count", "total_charges", "monthly_charges", "phone_service"],axis=1)

###########################
# categorize tenure months into years
        tenure_years = pd.cut(df['tenure_months'],
                                bins=[0,12,24,36,48,60,99],
                                labels=[1,2,3,4,5,6]
                                )
        tenure_years.value_counts(dropna=False)
        
        df_dummies = pd.get_dummies(tenure_years, prefix = "tenure")
        df_dummies
        
        # Combine
        finaldf = pd.merge(df, df_dummies, left_index=True, right_index=True)
        finaldf = finaldf.drop(["tenure_months"],axis=1)
        df=finaldf.copy()
        del(finaldf)

###########################
# Random select 10% sample

        df_sample=df.sample(frac = 0.2)
        
        df_sample['id'] = np.arange(len(df_sample))+1
        
        df_sample = df_sample.drop(["customerid"],axis=1)
        
        
###########################
# Graph

df_long = df_sample.drop(["churn_label"],axis=1)
df_long = df_sample.melt(id_vars=["id"])
df_long = df_long.sort_values(by=["id"])
df_long

df_graph = (ggplot(df_long, aes(x="id", y="variable", fill="value"))
 + geom_tile()
 + scale_fill_gradient(low = "gray", high = "black")
 + xlab("ID")
 + theme(
        axis_text_x = element_blank(),
        axis_title_y = element_blank(),
        legend_position="none",
        legend_title=element_blank(),
        panel_grid = element_blank()
)
)

ggsave(plot = df_graph, filename = "graph_customers.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 6)

###########################
# Within Clusters Sum of Squares(WCSS) 

df_segm = df_sample.drop(["id","churn_label"],axis=1)

wcss = []
for i in range(1,10):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(df_segm)
    wcss.append(kmeans_pca.inertia_)

        plt.clf() # this clears the figure
        plt.figure(figsize = (10,8))
        plt.plot(wcss, marker = 'o', linestyle = '-.',color='red')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('K-means Clustering')
        plt.show()

df_wcss = pd.DataFrame(wcss, columns=["wcss"])
df_wcss["delta"] = df_wcss.pct_change()
df_wcss["diff"] = df_wcss["delta"].diff()
df_wcss

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans.fit(df_segm)
df_sample["segment"] = kmeans.labels_


###########################
# Graph by segment

        df_sample["segment"].value_counts(dropna=False).sort_values()

        df_segm_long = df_sample.drop(["churn_label"],axis=1)
        df_segm_long = df_segm_long.melt(id_vars=["segment","id"])
        df_segm_long = df_segm_long.sort_values(by=["segment","id"])
        df_segm_long
        
        # create sequential, unique id within each segment to make nicer graph
        df_unique = df_segm_long.groupby(["segment","id"]).take([0]).reset_index()
        df_unique["group"] = df_unique.groupby(['segment']).cumcount()
        df_unique = df_unique.drop(columns=["variable","value","level_2"])
        df_segm_long = pd.merge(df_segm_long, df_unique, on=["segment","id"])

        df_segm_long
        
        df_graph = (
                ggplot(df_segm_long, aes(x='group', y="variable", fill="value"))
                + geom_tile()
                + facet_wrap("~segment",nrow=1, scales="free_x")
                + scale_fill_gradient(low = "gray", high = "black")
                + xlab("ID")
                + theme(
                        axis_text_x = element_blank(),
                        axis_title_y = element_blank(),
                        legend_position="none",
                        legend_title=element_blank(),
                        panel_grid = element_blank()
                )
                )

        df_graph

        ggsave(plot = df_graph, filename = "graph_customers_cluster.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 6)

###########################
# Segment descriptives

# plot segment by percent
        df_segment = pd.DataFrame(df_sample["segment"].value_counts(normalize=True).sort_values())
        df_segment['number'] = df_segment.index
        df_segment
        
        df_graph = (ggplot(df_segment, aes(x="number",y="segment", label="segment"))
         + geom_bar(stat="identity")
         + geom_text(va="bottom", format_string='{:.2f}')
         + theme(
                axis_title_x = element_blank(),
                axis_title_y = element_blank(),
                legend_position="none"
        )
        )
        df_graph
        
        ggsave(plot = df_graph, filename = "graph_segment.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)


## plot churn by segment

        df_segment = df_sample.groupby('segment')['churn_label'].mean().reset_index()
        df_segment
        
        df_graph = (ggplot(df_segment, aes(x="segment",y="churn_label", label="churn_label"))
         + geom_bar(stat="identity")
         + geom_text(va="bottom", format_string='{:.2f}')
         + theme(
                axis_title_x = element_blank(),
                axis_title_y = element_blank(),
                legend_position="none"
        )
        )
        df_graph
        
        ggsave(plot = df_graph, filename = "graph_segment_churn.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

        
