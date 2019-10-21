# IMPORTS  ==============================================
# ml test scripts
# silence warning output from scikit's FutureWarning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# import data set & model
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# PROGRAM  ==============================================
# load data sets
# rem 'target' contains labels that model will use to learn
iris = load_iris()  # the training data set
model = LogisticRegression().fit(iris['data'], iris['target'])

# use the stored model to make predictions
print(model.predict(iris['data']))

'''
rem: in scikit-learn, every algorithm follows
the API where you call .fit() to build your model or fit your model
and then .predict to make predictions using that stored model.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#%matplotlib inline

titanic = pd.read_csv(r'C:\Users\stacy\My GitHub\Machine-Learning-Miscl\titanic.csv')
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
titanic.drop(cat_feat, axis = 1, inplace = True)
result = titanic.head()
print(result)
