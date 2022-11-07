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
# Load data

# df = pd.read_csv(os.path.join(main_dir,data_files,"churn_model.csv"), index_col=0)
# df["churn_label"].value_counts(dropna=False)
# X = df.drop(["churn_label", "customerid", "total_charges", "monthly_charges", "phone_service"],axis=1)


dataframe   = ({
        'id':[1, 2, 3, 4],
        'v1' :[1, 1, 0, 0],
        'v2' :[1, 1, 0, 0],
        'v3' :[1, 0, 1, 0],
        'v4' :[1, 0, 1, 0]
        })

df = pd.DataFrame(dataframe, columns=['id','v1','v2','v3',"v4"])
df = df.sort_values(by=['id'])

df
df_long = df.melt(id_vars=["id"])
df_long

df_long["value"].value_counts(dropna=False)


(ggplot(df_long, aes(x="id", y="variable", fill="value"))
 + geom_tile(color="white", size=0.5)
 + coord_flip()
 + theme_bw()
 + scale_fill_gradient(low = "gray", high = "black")
 + theme(
        axis_title_x = element_blank(),
        axis_title_y = element_blank(),
        legend_position="none",
        legend_title=element_blank(),
        panel_grid = element_blank()
)
)

###########################
# Within Clusters Sum of Squares(WCSS) 

wcss = []
for i in range(1,5):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(df)
    wcss.append(kmeans_pca.inertia_)

        plt.clf() # this clears the figure
        plt.figure(figsize = (10,8))
        plt.plot(wcss, marker = 'o', linestyle = '-.',color='red')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('K-means Clustering')
        plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
kmeans.fit(df)
df["Segment K-means"] = kmeans.labels_
df

df_segm_analysis = df.groupby(['Segment K-means']).mean()
df_segm_analysis


