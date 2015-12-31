
# coding: utf-8

# In[2]:

from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")
data_all = pd.concat([data_train, data_test])


# In[3]:

print "Missing data train set:"
miss = len(data_train.index) - data_train.count()
print 1. * miss / len(data_train.index) 


# In[4]:

print "Missing data test set:"
miss = len(data_test.index) - data_test.count()
print 1. * miss / len(data_test.index)


# In[5]:

# should create multiple data frames to have multiple plots
survived = pd.DataFrame( data_train.Survived.value_counts() )
survived.index = ["Died", "Survived"]
survived.plot(kind="bar", rot=0, title="Distribution of survived passagers")

pclass = pd.DataFrame( data_train.Pclass.value_counts() )
pclass.plot(kind="bar", rot=0, title="Distribution by class")

sex = pd.DataFrame( data_train.Sex.value_counts() )
sex.plot(kind="bar", rot=0, title="Distribution by sex")

d = data_train.groupby(["Pclass", "Survived"]).size()
d = d.unstack()
d.plot(kind="bar", rot=0, stacked=True)


# In[6]:

# Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 2 NA
s = data_train.Embarked
s.fillna(s.value_counts().idxmax(), inplace=True)
embarked = pd.DataFrame( data_train.Embarked.value_counts() )
embarked.plot(kind="bar", rot=0, title="Distribution by place of embarkation")


# In[7]:

# feature distribution + survival

d = data_train.groupby(["Pclass", "Survived"]).size()
d = d.unstack()
d.plot(kind="bar", rot=0, stacked=True)

d = data_train.groupby(["Sex", "Survived"]).size()
d = d.unstack()
d.plot(kind="bar", rot=0, stacked=True)

d = data_train.groupby(["Embarked", "Survived"]).size()
d = d.unstack()
d.plot(kind="bar", rot=0, stacked=True)


# In[8]:

# extracting title

data_train["Title"] = data_train.Name.str.extract(".*, ([a-zA-Z ]*)\. .*")
data_test["Title"] = data_test.Name.str.extract(".*, ([a-zA-Z ]*)\. .*")

print "Titles:"
print data_train.Title.value_counts()

# want to preserve column order
d = OrderedDict()
d["Count"] = len
d["Missing"] = lambda x: x.isnull().sum()
d["Mean"] = np.mean
#print
#print data_train.groupby("Title")["Age"].agg(d)

similar = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"]
data_train.Title.replace(similar, "Aristocratic", inplace=True)
data_train.Title.replace("Ms", "Mrs", inplace=True)
data_train.Title.replace(["Mlle", "Mme"], "Miss", inplace=True)

data_test.Title.replace(similar, "Aristocratic", inplace=True)
data_test.Title.replace("Ms", "Mrs", inplace=True)
data_test["Title"].replace(["Mlle", "Mme"], "Miss", inplace=True)


# In[9]:

from sklearn.preprocessing import LabelEncoder
def add_cat(data, col):
    enc = LabelEncoder()
    enc.fit(data[col])
    data[col + "Cat"] = enc.transform(data[col])
    
add_cat(data_train, "Title")
add_cat(data_test, "Title")


# In[10]:

# fill age na values with means from Title
def func(x):
    x.fillna(x.mean(), inplace=True)
    # have to replace even if transform.. somehow
    return x

data_train.Age = data_train.groupby("Title").Age.apply(func)
data_test.Age = data_test.groupby("Title").Age.apply(func)

# fill zero fare
data_train.Fare.replace(0, None, inplace=True)
data_train.Fare = data_train.groupby("Title").Fare.apply(func)

data_test.Fare.replace(0, None, inplace=True)
data_test.Fare = data_test.groupby("Title").Fare.apply(func)

# at this point should not have 0 or NA fare
# make check
incorrect = lambda data: len(data) - data.Fare.count() != 0 or (data.Fare == 0).any()
if incorrect(data_train) or incorrect(data_test): raise Exception("data train or test Fare NA or 0 values")


# In[11]:

dd = pd.DataFrame(data_train[["Age", "Title"]])
dd.boxplot(column="Age", by="Title")


# In[12]:

# alive
p = data_train.groupby("Title")["Survived"].agg(lambda x: 1. * x.sum()/len(x)).plot(kind="bar")
p.set_ylim([0,1])
p


# In[13]:

# this one i should change a bit
data_train['PclassCat'] = data_train['Pclass'].astype('category')
ax = data_train.loc[data_train['Survived'] == 1].plot(kind='scatter', x='Age', y='Pclass', color='DarkBlue', label='Survived')
data_train.loc[data_train['Survived'] == 0].plot(kind='scatter', x='Age', y='Pclass', color='DarkGreen', label='Died', ax=ax)


# In[14]:

miss = len(data_train.index) - data_train.count()
h = len(miss > 0)
print
if h > 0:
    print "Missing data train set:"
    print miss[miss > 0]
else:
    print "No missing data in train set"


# In[15]:

miss = len(data_test.index) - data_test.count()
h = len(miss > 0)
print
if h > 0:
    print "Missing data test set:"
    print miss[miss > 0]
else:
    print "No missing data in test set"


# In[16]:

# add IsCabin feature
data_train['IsCabin'] = 0
data_train.loc[data_train.Cabin.notnull(), 'IsCabin'] = 1

data_test['IsCabin'] = 0
data_test.loc[data_test.Cabin.notnull(), 'IsCabin'] = 1


# In[17]:

print len(data_train[data_train.Fare.isnull()])
print len(data_test[data_test.Fare.isnull()])


# In[18]:

# find correlation between survived and fare + woman or man
print data_train[['Fare', 'Survived']].corr()


# In[19]:

yo = data_train.groupby(['Title', 'Pclass', 'Survived']).size().unstack()
yo.fillna(0, inplace=True)
print yo


# In[34]:

# working on last name
data_all['LastName'] = data_all.Name.str.extract("(.+),.+")
data_train['LastName'] = data_train.Name.str.extract("(.+),.+")
data_test['LastName'] = data_train.Name.str.extract("(.+),.+")

# find last names that survived

#from sets import Set 
s = data_train[data_train.Survived == 1].LastName.unique()
survived_lastnames = set(s)

sz = data_all.groupby('LastName', group_keys='lol').size()
for (index, value) in sz.iteritems():
    if value == 1: survived_lastnames.discard(value)

# should be 3 variants 
cond = lambda x: 1 if x in survived_lastnames else 0
data_train['RelativeSurvived'] = data_train['LastName'].apply(cond)
data_test['RelativeSurvived'] = data_test['LastName'].apply(cond)

print "Do we have null values somewhere in LastName"
print data_all.LastName.isnull().any()


# In[ ]:



