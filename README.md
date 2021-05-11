# Logistic-Regression-using-Python
# Titanic dataset
# Classification / binary / dichotomous / discrete / attributes / probability
import os #operating system
os.getcwd()
os.chdir("C:\\Users\\Arjun\\Desktop\\Python\\LogitRegression")
import numpy as np #array and mathematical calculation
import pandas as pd #import, export, manipulation
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns #stats and vis
ship = pd.read_csv("titanic.csv")
ship.head()
# see the distinct values in the survived column
ship.Survived.value_counts()
df.Sex.value_counts()
df.columns
Z = pd.DataFrame()
Z['pclass'] = ship['Pclass']
Z['sex'] = ship['Sex']
Z['age'] = ship['Age']
Z['sibsp']= ship['SibSp']
Z['parch'] = ship['Parch']
Z['embarked']= ship['Embarked']
# these variable are called Idependent Variable
Z.head()
y = ship['Survived']
y
## Checking Missing Values in Titanic Dataset
df.isnull().sum()
Z.head(2)
Z.hist('age')
sns.boxplot(y='age', data=Z)
#Outlier found in Age so we shall use median to fill the missing values
Z['age'] = Z['age'].fillna(Z.age.median())
print(Z.age.isnull().sum())
Z.info()
Z['embarked'] = Z['embarked'].fillna(Z.embarked.mode()[0])
print(Z.embarked.isnull().sum())
print(Z.isnull().sum())
Z.info()
#Encoding
print(Z.sex[:5])
Z['sex'] = pd.get_dummies(Z.sex)['female']
print(Z.sex[:5])
Z.head()
Z = Z.join(pd.get_dummies(df.Embarked, prefix='Embarked'))
display(Z[:5])
Z = Z.drop(['embarked','Embarked_C'], axis = 1)
Z.head()
display(Z[:5])
Z = Z.join(pd.get_dummies(df.Pclass, prefix='Pclass'))
display(Z[:5])
display(Z[:5])
Z = Z.drop(['Pclass_1','Pclass_2','Pclass_3'],axis=1)
display(Z[:5])
# Standarisation with age variable
# sklearn - machine learning package
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
display(Z[:5])
Z.age = scaler.fit_transform(Z[['age']])
display(Z[:5])
#Model Building
#split the data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size=0.20,random_state= 101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# Building Logistic Regression Model with training dataset 
from sklearn.linear_model import LogisticRegression
# fit the model to the training dataset
model = LogisticRegression()
model.fit(x_train,y_train)
print(model.intercept_)
print(model.coef_)
print(x_train.columns)
# Predict the model with test dataset
display(x_test[:10])
display(model.predict_proba(x_test)[:10])# probability
print()
display(model.predict(x_test)[:10]) # classification prediction
y_pred= model.predict(x_test)
from sklearn.metrics import roc_auc_score
# lets measure the logistic model with ROC and AUC curve
logistic_roc_auc = roc_auc_score(y_test,model.predict(x_test) )
logistic_roc_auc
# K-Fold Methods
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = model, X = x_train, y=y_train, cv=15)
accuracy
accuracy[6]
accuracy.mean()



































