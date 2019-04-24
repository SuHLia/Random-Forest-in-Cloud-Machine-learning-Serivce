#!/usr/bin/env python
# coding: utf-8

# import the libraies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Set the style
plt.style.use('fivethirtyeight')


# load the dataset 
df = pd.read_csv("s3://finalprojectsh/glass.csv")


# get an idea of the dataset by using head()
df.head()


# check the data type of each attribute
df.dtypes


# check the dataset size
df.shape



# use descriptive statistic analysis to explore the data
df.describe()


# check the distinct types of glass
print("The total distinct types of glass:", df["Type"].nunique())


# explore the relation between chemical elements w.r.t the type of glass
sns.pairplot(df, kind="scatter", hue="Type", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()



# set up the features and label
y = df["Type"]
X = df.drop("Type", axis = 1)


# split the training data and test data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 0)
print('Training Features Shape:', train_X.shape)
print('Training Label Shape:', train_y.shape)
print('Testing Features Shape:', test_X.shape)
print('Testing Label Shape:', test_y.shape)


# use the training dataset to train the model and use test dataset to do prediction
rf = RandomForestClassifier(n_estimators=20,random_state=0)
rf.fit(train_X, train_y)
y_pred = rf.predict(test_X)
print("Confusion Matrix:", confusion_matrix(test_y, y_pred), sep="\n")
print("\n Accuracy Score:", round(accuracy_score(test_y,y_pred)*100, 2), '%.')


# create an importance Matrix
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = rf.feature_importances_
features.sort_values(by=['importance'], ascending = False, inplace = True)
print("Feature importance Matix:",features, sep = "\n")

# visualize the feature importance
sns.set(style="whitegrid")
ax = sns.barplot(x="feature", y="importance", data=features, palette="Blues_d")
ax.set_title('Features Importance')
plt.show()





