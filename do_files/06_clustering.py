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

df = pd.read_csv(os.path.join(main_dir,data_files,"churn_model.csv"), index_col=0)
df.columns.sort_values()

# drop phone_service, monthly_charges, total_charges, tenure
df = df[[
        'churn_label', 
        'contract_month_to_month','contract_one_year', 'contract_two_year', 
        'senior_citizen','dependents','gender','partner',
        'device_protection_yes',
        "internet_service_dsl", 'internet_service_fiber_optic', 'internet_service_no',
        'multiple_lines_yes',
        'online_backup_yes',
        'online_security_yes',
        'paperless_billing',
        'payment_bank_transfer_auto','payment_credit_card', 'payment_electronic_check','payment_mailed_check',
        'streaming_movies_yes',
        'streaming_tv_yes',
        'tech_support_yes'
        # 'tenure_1','tenure_2', 'tenure_3', 'tenure_4', 'tenure_5', 'tenure_6',
       ]]

###########################
# Random select 10% sample

        df_sample=df.copy()
        df_sample=df.sample(frac = 0.2)

        df_sample['id'] = np.arange(len(df_sample))+1
        
###########################
# Graph

df_long = df_sample.drop(["churn_label"],axis=1)
df_long = df_long.melt(id_vars=["id"])
df_long = df_long.sort_values(by=["id"])
df_long["variable"].value_counts().sort_values()

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

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
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

        
