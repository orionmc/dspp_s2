# imports
import csv
import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")


#pd.set_option("display.max_rows",None)
#matplotlib.style.use('ggplot')

# Load the data & various visualizations & checks
heartds=pd.read_csv('C:\\Users\\user0\\Documents\\BPP\\DS-Proffessional-Practice\\_DataSets\\Heart-Attack\\heart0.csv')
heartds.head()
heartds.info()
heartds[heartds.duplicated()]
heartds.describe()
# heartds.describe().T # matrix rotated to 90 degrees

# pick suitable colour for the plot and draw the correlation plot

px.imshow(heartds.corr(),title="Correlation Matrix",color_continuous_scale="viridis")




# Graph distribution & linearity of the variables
"""
plt.figure(figsize=(15,10))
for i,col in enumerate(heartds.columns,1):
    plt.subplot(4,3,i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(heartds[col],kde=True)
    plt.tight_layout()
    plt.plot()

# Numeric & Categorical Variables
numeric_var = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
categoric_var = ["sex", "cp", "fbs", "rest_ecg", "exng", "slpe", "caa", "thall", "target"]

numeric_axis_name = ["Age of the patient", "Resting Blood Pressure", "Cholestrol", "Max HR", "Oldpeak"]
zipped_list_for_numeric = list(zip(numeric_var, numeric_axis_name))

categoric_axis_name = ["Gender", "Chest Pain Type", "Fasting Blood Sugar", "Resting Electrocardiographic Results",
                       "Exercise Induced Angina", "Slope of the ST Segment", "Number of Major Vessels", "Thall", "Target"]
zipped_list_for_categoric = list(zip(categoric_var, categoric_axis_name))

title_font = {"family": "arial", "color": "darkred", "weight": "bold", "size": 15}
axis_font = {"family": "arial", "color": "darkblue", "weight": "bold", "size": 13}


for i,z in zipped_list_for_numeric:
    graph = sns.FacetGrid(heartds[numeric_var], hue = 'target', height = 5, xlim = ((heartds[i].min() - 10), (heartds[i].max() + 10)))
    graph.map(sns.kdeplot, i, shade = True)
    graph.add_legend()
    
    plt.title(i, fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Density", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()

"""