###########################
# TOP COMMANDS
###########################
# create empty session
globals().clear()

# load libraries
import os
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats import ttest_ind_from_stats

# beginning commands
np.set_printoptions(suppress=True) # drop scientific notation
pd.set_option('display.max_columns', None) # display max columns

# file paths - adapt main_dir pathway
main_dir = "/Users/jonathanlatner/GitHub/ab_test/"
data_files = "data_files/"
graphs = "graphs/"
tables = "tables/"

###########################
# LOAD DATA
###########################

df = pd.read_csv(os.path.join(main_dir,data_files,"ab_test.csv"), index_col=0)

###########################
# DESCRIPTIVES
###########################

df.columns
df.describe()
df.info()

df["treat"].value_counts(dropna=False).sort_index()
df['sessions'].sum()

# print head into .tex
# df.head(10).style.format(precision=1).to_latex()
df.head(10).style.to_latex(os.path.join(main_dir,tables,"data_frame.tex"))

# Rename values within device category using dictionary
a = ["desktop", "mobile", "tablet"]
b = ["D", "M", "T"]
df['device'] = df['device'].map(dict(zip(a, b)))

# Categorize days into weeks
df["week"] = pd.cut(df["date"],bins=[0,20190108,20190115,20190122,20190132],labels=[1,2,3,4])

###########################
# Graph total values
###########################

# aggregate
df_long = df.drop(columns=["date","week","device"])
df_long = df_long.melt(id_vars=['treat'])
df_long = df_long.groupby(['treat',"variable"])[["value"]].sum().reset_index()
df_long["value"]=df_long["value"]/1000
df_long

df_graph = (ggplot(df_long, aes(x = 'treat', y = "value", fill = "treat", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
+ geom_text(format_string='{:,.2f}', va="top", position = position_dodge(width = 0.9))
+ facet_wrap('~variable',scales="free")
+ ylab("Values (1000s)")
+ theme(
        legend_title=element_blank(),
        legend_position="none",
        axis_title_x = element_blank(),
        subplots_adjust={'wspace':0.3, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_values.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

# by device

df_long = df.melt(id_vars=['treat',"device","date","week"])
df_long = df_long.groupby(['treat',"device","variable"])[["value"]].sum().reset_index()
df_long["value"]=df_long["value"]/1000
df_long

df_graph = (ggplot(df_long, aes(x = 'device', y = "value", fill = "treat", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
# + geom_text(format_string='{:,.2f}', va="top", position = position_dodge(width = 0.9))
+ facet_wrap('~variable',scales="free")
+ ylab("Values (1000s)")
+ theme(
        legend_title=element_blank(),
        legend_position="bottom",
        # axis_title_x = element_blank(),
        legend_box_spacing=.25,
        subplots_adjust={'wspace':0.3, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_values_device.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

# by week
df_long = df.melt(id_vars=['treat',"device","date","week"])
df_long = df_long.groupby(['treat',"variable","week"])[["value"]].sum().reset_index()
df_long["value"]=df_long["value"]/1000
df_long

df_graph = (ggplot(df_long, aes(x = 'week', y = "value", fill = "treat", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
# + geom_text(format_string='{:,.2f}', va="top", position = position_dodge(width = 0.9))
+ facet_wrap('~variable',scales="free", ncol=3)
+ ylab("Values (1000s)")
+ theme(
        legend_title=element_blank(),
        legend_position="bottom",
        legend_box_spacing=.25,
        # axis_title_x = element_blank(),
        subplots_adjust={'wspace':0.3, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_values_time.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# Calculate difference by device
###########################

###########################
# aggregate
df_long = df.melt(id_vars=['treat',"device","date","week"])
df_long = df_long.groupby(['treat',"variable"])[["value"]].sum().reset_index()
df_long

# calculate difference
df_long["lag_value"] = df_long.groupby(['variable'])['value'].shift(1)
df_long["difference"] = (df_long["value"]-df_long["lag_value"])/df_long["lag_value"]*100

# drop, rename, and filter 
df_long=df_long.drop(columns=["lag_value","value"])
df_long=df_long.rename(columns={"difference":"value"})
df_long=df_long.dropna(subset=['value'])
df_long["positive"] = np.where(df_long["value"]>0, df_long["value"], np.nan)
df_long["negative"] = np.where(df_long["value"]<0, df_long["value"], np.nan)

# graph
df_graph = (ggplot(df_long, aes(x = 'variable', y = "value", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
+ geom_text(aes(y="positive"),
                format_string='{:,.2f}', 
                va="bottom",
                position = position_dodge(width = 0.9))
+ geom_text(aes(y="negative"),
                format_string='{:,.2f}', 
                va="top",
                position = position_dodge(width = 0.9))
# + facet_wrap('~variable',nrow=1)
+ ylab("% Difference")
+ theme(
        legend_title=element_blank(),
        legend_position="bottom",
        axis_title_x = element_blank(),
        subplots_adjust={'wspace':0.5, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_diff.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)


###########################
# by device
df_long = df.drop(columns=["date","week"])
df_long = df_long.melt(id_vars=['treat',"device"])
df_long = df_long.groupby(['treat',"device","variable"])[["value"]].sum().reset_index()
df_long

# calculate difference
df_long["lag_value"] = df_long.groupby(['variable',"device"])['value'].shift(1)
df_long["difference"] = (df_long["value"]-df_long["lag_value"])/df_long["lag_value"]*100

# drop, rename, and filter 
df_long=df_long.drop(columns=["lag_value","value"])
df_long=df_long.rename(columns={"difference":"value"})
df_long=df_long.dropna(subset=['value'])
df_long["positive"] = np.where(df_long["value"]>0, df_long["value"], np.nan)
df_long["negative"] = np.where(df_long["value"]<0, df_long["value"], np.nan)
df_long

# graph
df_graph = (ggplot(df_long, aes(x = 'device', y = "value", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
+ geom_text(aes(y="positive"),
                format_string='{:,.2f}', 
                va="bottom",
                position = position_dodge(width = 0.9), 
                size=7.5)
+ geom_text(aes(y="negative"),
                format_string='{:,.2f}', 
                va="top",
                position = position_dodge(width = 0.9), 
                size=7.5)
+ facet_wrap('~variable',nrow=1)
+ ylab("% Difference")
+ theme(
        legend_title=element_blank(),
        legend_position="bottom",
        axis_title_x = element_blank(),
        # subplots_adjust={'wspace':0.5, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_diff_device.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)

###########################
# by week
df_long = df.drop(columns=["date","device"])
df_long = df_long.melt(id_vars=['treat',"week"])
df_long = df_long.groupby(['treat',"week","variable"])[["value"]].sum().reset_index()
df_long

# calculate difference
df_long["lag_value"] = df_long.groupby(['variable',"week"])['value'].shift(1)
df_long["difference"] = (df_long["value"]-df_long["lag_value"])/df_long["lag_value"]*100

# drop, rename, and filter 
df_long=df_long.drop(columns=["lag_value","value"])
df_long=df_long.rename(columns={"difference":"value"})
df_long=df_long.dropna(subset=['value'])
df_long["treat"] = " Difference"
df_long["positive"] = np.where(df_long["value"]>0, df_long["value"], np.nan)
df_long["negative"] = np.where(df_long["value"]<0, df_long["value"], np.nan)

# graph
df_graph = (ggplot(df_long, aes(x = 'week', y = "value", label = "value"))
+ geom_bar(stat="identity", position = position_dodge(width = 0.9))   
+ geom_text(aes(y="positive"),
                format_string='{:,.2f}', 
                va="bottom",
                position = position_dodge(width = 0.9), 
                size=7.5)
+ geom_text(aes(y="negative"),
                format_string='{:,.2f}', 
                va="top",
                position = position_dodge(width = 0.9), 
                size=7.5)
+ facet_wrap('~variable',nrow=1)
+ ylab("% Difference")
+ theme(
        legend_title=element_blank(),
        legend_position="bottom",
        axis_title_x = element_blank(),
        # subplots_adjust={'wspace':0.5, 'hspace':0.5}
        )
)

df_graph

ggsave(plot = df_graph, filename = "graph_diff_time.pdf", path = os.path.join(main_dir,graphs), width = 10, height = 4)
