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
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
warnings.filterwarnings("ignore")


 #pd.set_option("display.max_rows",None)
 #matplotlib.style.use('ggplot')

 # Load the data & various visualizations & checks
heartds=pd.read_csv('C:\\Users\\user0\\Documents\\BPP\\DS-Proffessional-Practice\\_DataSets\\Heart-Attack\\heart0.csv')
heartds.head()
heartds.info()
heartds[heartds.duplicated()]
heartds.describe()
# heartds.describe().T # matrix rotated 90 degrees

# pick suitable colour for the plot and draw the correlation plot
fig = px.imshow(heartds.corr(),title="Correlation Matrix",color_continuous_scale="viridis")
fig.show()

# Numerical correlation matrix
numeric_corr_matrix = heartds.corr()
print(numeric_corr_matrix.to_string())


# Graph distribution & linearity of the variables
plt.figure(figsize=(15,10))
for i,col in enumerate(heartds.columns[:-2],1):
    plt.subplot(4,3,i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(heartds[col],kde=True)
    plt.tight_layout()
    plt.plot()

# Numeric & Categorical Variables
numeric_var = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
categoric_var = ["sex", "cp", "fbs", "rest_ecg", "exng", "slp", "caa", "thall", "output"]
numeric_var.append("output")


numeric_axis_name = ["Age of the patient", "Resting Blood Pressure", "Cholestrol", "Max HR", "Oldpeak"]
zipped_list_for_numeric = list(zip(numeric_var, numeric_axis_name))

categoric_axis_name = ["Gender", "Chest Pain Type", "Fasting Blood Sugar", "Resting Electrocardiographic Results",
                       "Exercise Induced Angina", "Slope of the ST Segment", "Number of Major Vessels", "Thall", "Output"]
zipped_list_for_categoric = list(zip(categoric_var, categoric_axis_name))

title_font = {"family": "arial", "color": "darkred", "weight": "bold", "size": 15}
axis_font = {"family": "arial", "color": "darkblue", "weight": "bold", "size": 13}

# Draing graphs for numeric variables
for i,z in zipped_list_for_numeric:
    fig = sns.FacetGrid(heartds[numeric_var], hue = 'output', height = 5, xlim = ((heartds[i].min() - 10), (heartds[i].max() + 10)))
    fig.map(sns.kdeplot, i, shade = True)
    #graph.map(sns.histplot, i, bins=10) # different visualisation if needed
    fig.add_legend()
    
    plt.title(i, fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Density", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()
# Drawing graphs for categorical variables
for i,z in zipped_list_for_categoric:
    plt.figure(figsize = (8, 5))
    sns.countplot(x = i, data = heartds[categoric_var], hue = "output")
    
    plt.title(i + " - output", fontdict = title_font)
    plt.xlabel(z, fontdict = axis_font)
    plt.ylabel("Output", fontdict = axis_font)
    
    plt.tight_layout()
    plt.show()

# Data Preprocessing
x = heartds.iloc[:, 1:-1].values
y = heartds.iloc[:, -1].values
# print(x,y)

# Splitting the dataset into training and testing dataset 20%-80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# x_train,x_test

# Support Vector Classification
model = SVC()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("The accuracy of SVM algorithm is: ", accuracy_score(y_test, predicted)*100, "%")

# # Random Forest
model = RandomForestRegressor(n_estimators = 100, random_state = 0)  
model.fit(x_train, y_train)  
predicted = model.predict(x_test)
print("The accuracy of Random Forest algorithm is : ", accuracy_score(y_test, predicted.round())*100, "%")












