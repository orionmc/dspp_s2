import csv
import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from sklearn import preprocessing
import matplotlib


warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder


# Load the data & various visualizations & checks
heartds=pd.read_csv('C:\\Users\\user0\\Documents\\BPP\\DS-Proffessional-Practice\\_DataSets\\Heart-Attack\\heart0.csv')
heartds.head()
heartds.info()
heartds[heartds.duplicated()]
heartds.describe()
# heartds.describe().T # matrix rotated to 90 degrees

# pick suitable colour for the plot and draw the correlation plot
px.imshow(heartds.corr(),title="Correlation Plot",color_continuous_scale="viridis")

