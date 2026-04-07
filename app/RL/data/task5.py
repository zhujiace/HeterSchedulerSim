#
# Copyright 2023 The EHPCL Plotting Authors.
#

# import necessary libs

import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

# import pandas as pd
# import numpy as np
SAVEPATH = "./"

# load data
# can also use numpy or pandas dataframe
import pandas as pd
# rrw = pd.read_csv('../data/rr_task.csv')
# edf = pd.read_csv('../data/edf_dp_task.csv')
# shape = pd.read_csv('../data/shape_task.csv')
# ss = pd.read_csv('../data/ss_task5.csv')
edf = pd.read_csv('edf_results.csv')
dy = edf.groupby('uti')['schedulable'].mean()
# dx = [1.1, 1.2, 1.3, 1.4, 1.5]
# dy1 = [4, 5, 7.5, 8, 10]
# dy2 = [6.5, 8.5, 11.5, 12, 14]
# dy3 = [8, 6.5, 9, 11, 12]
dx = [round(2.5 + i * 0.1, 1) for i in range(15)]


# setting ploting size, linewidth, fonts
sns.set_context('talk')
sns.plotting_context()
sns.set_context('talk', rc={'lines.linewidth': 1.875})
# ubuntu does not have times new roman by default
# need to install times new roman
# ask chatGPT if you want to use times new roman
# plt.rcParams['font.family'] = 'Times New Roman'

# get figure axes
fig = plt.figure(figsize=(10,6), edgecolor='black')
ax1: Axes = plt.axes()
plt.grid(True, linestyle='-.', fillstyle='left', alpha=0.9)

# plot on the axes

tasknum = 5
# dx = shape[shape["task"]==tasknum]["uti"]
# dy = shape[shape["task"]==tasknum]["accept"]/10
ax1.plot(dx, dy, 's-',
     color="#6c8ebf",
     markeredgecolor="black",
     markerfacecolor=(218/255,232/255,252/255),
     label="SHAPE",
     linewidth=3,
     markersize=12)

# dx = ss[ss["task"]==tasknum]["uti"]/100
# dy = ss[ss["task"]==tasknum]["accept"]/10
ax1.plot(dx, dy, 'D-',
     color="#d6b656",
     markeredgecolor="black",
     markerfacecolor='#fff2cc',
     label="SelfS",
     linewidth=3,
     markersize=10)

# dx = edf[edf["task"]==tasknum]["uti"]
# dy = edf[edf["task"]==tasknum]["accept"]/10
# ax1.plot(dx, dy, 'p-',
#      color="#82b366",
#      markeredgecolor="black",
#      markerfacecolor='#b5dba2',#97d077
#      label="EDF-L ",
#      linewidth=3,
#      markersize=14)

# dx = rrw[rrw["task"]==tasknum]["utilization"]
# dy = rrw[rrw["task"]==tasknum]["accept"]/10
# ax1.plot(dx, dy, 'h-',
#      color='#b85450',
#      markeredgecolor="black",
#      markerfacecolor=(248/255,206/255,204/255),#97d077
#      label="HARD ",
#      linewidth=3,
#      markersize=14)


# Set the usetex parameter to True
# Need to intall latex in the path
# sudo apt-get install texlive-full
plt.rcParams['text.usetex'] = True
plt.xlabel("Utilization", fontsize=24)
plt.ylabel("Schedulable Ratio (%)", fontsize=24)
plt.legend(
           fontsize=20, ncol=1,
           loc='upper right')
plt.xlim(1.05, 2.65)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

# do remember to save figures
# recommended file type: pdf
# use "bbox_inches='tight'" to cut extra white edges
plt.savefig(os.path.join(SAVEPATH, "EDF_DP_TASK5.pdf"), bbox_inches='tight')
plt.show()
plt.savefig()
