
# coding: utf-8

# In[3]:

from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")
data_all = pd.concat([data_train, data_test])


# after concatenation easy to pick train vs test by Survived null column


# In[4]:

print("===== survived by class and sex")
print(data_train.groupby(["Pclass", "Sex"])["Survived"].value_counts(normalize=True))


# In[5]:

describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

print("===== train: males")
print(data_train[data_train["Sex"] == "male"][describe_fields].describe())

print("===== test: males")
print(data_test[data_test["Sex"] == "male"][describe_fields].describe())

print("===== train: females")
print(data_train[data_train["Sex"] == "female"][describe_fields].describe())

print("===== test: females")
print(data_test[data_test["Sex"] == "female"][describe_fields].describe())


# In[6]:

data_all.sort_values(by=["Name"]).head(50)


# In[14]:

data_all['LastName'] = data_all.Name.str.extract("(.+),.+")
data_train['LastName'] = data_train.Name.str.extract("(.+),.+")


# In[15]:

data_train[(data_train.Parch + data_train.SibSp) >= 2].groupby(['LastName', 'Survived']).size()

# is it important that woman would be not alone ... family... 


# In[ ]:



