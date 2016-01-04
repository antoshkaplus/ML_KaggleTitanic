
# coding: utf-8

# In[133]:

from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")
data_all = pd.concat([data_train, data_test], ignore_index=True)


# In[134]:

print "Missing data train set:"
miss = len(data_train.index) - data_train.count()
print 1. * miss / len(data_train.index) 


# In[135]:

print "Missing data test set:"
miss = len(data_test.index) - data_test.count()
print 1. * miss / len(data_test.index)


# In[136]:

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


# In[137]:

# Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 2 NA
s = data_train.Embarked
s.fillna(s.value_counts().idxmax(), inplace=True)
embarked = pd.DataFrame( data_train.Embarked.value_counts() )
embarked.plot(kind="bar", rot=0, title="Distribution by place of embarkation")


# In[138]:

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


# In[139]:

pd.DataFrame(data_train.ix[data_train.Survived == 1, 'Age']).plot(kind='hist', bins=15)
pd.DataFrame(data_train.ix[data_train.Survived == 0, 'Age']).plot(kind='hist', bins=15)

pd.DataFrame(data_train.Age).plot(kind='hist', bins=15, alpha=0.5)

# and lets histograms by class
pd.DataFrame(data_all.ix[data_all.Pclass == 1, 'Age']).plot(kind='hist', bins=20)
pd.DataFrame(data_all.ix[data_all.Pclass == 2, 'Age']).plot(kind='hist', bins=20)
pd.DataFrame(data_all.ix[data_all.Pclass == 3, 'Age']).plot(kind='hist', bins=20)



# In[140]:

# extracting title

pattern = ".*, ([a-zA-Z ]*)\. .*"
data_train["Title"] = data_train.Name.str.extract(pattern)
data_test["Title"] = data_test.Name.str.extract(pattern)

data_all["Title"] = data_all.Name.str.extract(pattern)

print "Titles:"
print data_all.groupby(["Title"])['Age'].mean()

# want to preserve column order
d = OrderedDict()
d["Count"] = len
d["Missing"] = lambda x: x.isnull().sum()
d["Mean"] = np.mean
#print
#print data_train.groupby("Title")["Age"].agg(d)

# had to predict ages here!!!
# Mme = Missus (Mrs) 
# Mlle = Miss (no abbreviation, unless you use: Ms).

similar = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"]
data_train.Title.replace(similar, "Aristocratic", inplace=True)
data_train.Title.replace("Ms", "Mrs", inplace=True)
data_train.Title.replace(["Mlle", "Mme"], "Miss", inplace=True)

data_test.Title.replace(similar, "Aristocratic", inplace=True)
data_test.Title.replace("Ms", "Mrs", inplace=True)
data_test["Title"].replace(["Mlle", "Mme"], "Miss", inplace=True)

data_all.Title.replace(similar, "Aristocratic", inplace=True)
data_all.Title.replace("Mlle", "Miss", inplace=True)
data_all.Title.replace("Mme", "Mrs", inplace=True)
# no husband or relatives found for those ladyes
data_all.Title.replace("Ms", "Miss", inplace=True)

dd = pd.DataFrame(data_all[["Age", "Title"]])
dd.boxplot(column="Age", by="Title")


# In[141]:

from sklearn.preprocessing import LabelEncoder
def add_cat(data, col):
    enc = LabelEncoder()
    enc.fit(data[col])
    data[col + "Cat"] = enc.transform(data[col])
    
add_cat(data_train, "Title")
add_cat(data_test, "Title")


# In[142]:

print data_all.groupby(['Title', 'Pclass']).Age.agg([np.mean, np.size])

fillna_with_mean = lambda x: x.fillna(x.mean())

data_all.Age = data_all.groupby(['Title', 'Pclass']).Age.transform(fillna_with_mean)

data_train.Age = data_train.groupby("Title").Age.apply(func)
data_test.Age = data_test.groupby("Title").Age.apply(func)


# In[252]:

# lets see what we can get from fare
#print data_all.groupby(['Fare', 'LastName']).size()

data_all['Family'] = data_all.SibSp + data_all.Parch

# would start from 1 cause apply would use app twice on first group
family_id = 0
def app(group):
    global family_id
    sz = len(group)
    arr = None
    if group.Family.iloc[0] == 0:
        next_id = family_id + sz
        arr = range(family_id, next_id)
        family_id = next_id
    elif sz != group.Family.iloc[0] + 1:
        arr = sz*[None]
    else:
        arr = sz*[family_id]
        family_id += 1
    return pd.DataFrame({'FamilyId' : arr}, index=group.index)
    
def add_family_id(data):
    data['FamilyId'] = data.groupby(['LastName', 'Family']).apply(app).FamilyId
    
    
print len(data_all[data_all.FamilyId.isnull()])

#print list(data_all.index)
add_family_id(data_all)
print data_all




print data_all[data_all.Family > 0].groupby(['LastName', 'Family']).size()

# Thamine Thelma


print data_all.ix[data_all.Fare == 8.5167, [ 'Name', 'Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Thelma', [ 'Name', 'Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Thamine', [ 'Name', 'Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Thomas', [ 'Name', 'Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Allen', [ 'Name', 'Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Bowen', ['Name','Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Bradley', ['Name','Pclass', 'Fare', 'Parch', 'SibSp']]

print data_all.ix[data_all.LastName == 'Brown', ['Name','Pclass', 'Fare', 'Parch', 'SibSp']]





# family but different class
tt = data_all.groupby(['LastName']).Pclass.agg(lambda x: len(x.unique()) == 1)
print tt[tt == False]

print data_all.Fare.isnull().any()

print data_all[ data_all.LastName == data_all.ix[data_all.Fare.idxmax(), 'LastName'] ]

#probably we are going to categorize fare as  well, or maybe not.. we will see later

d = data_all[data_all.Fare < 100]
d = d[d.Fare < 30]
d.Fare.plot(kind='hist', bins=20)




print data_all.Fare.head()


# In[ ]:

# fill zero fare
data_train.Fare.replace(0, None, inplace=True)
data_train.Fare = data_train.groupby("Title").Fare.apply(func)

data_test.Fare.replace(0, None, inplace=True)
data_test.Fare = data_test.groupby("Title").Fare.apply(func)

# at this point should not have 0 or NA fare
# make check
incorrect = lambda data: len(data) - data.Fare.count() != 0 or (data.Fare == 0).any()
if incorrect(data_train) or incorrect(data_test): raise Exception("data train or test Fare NA or 0 values")


# In[ ]:

dd = pd.DataFrame(data_train[["Age", "Title"]])
dd.boxplot(column="Age", by="Title")


# In[ ]:

# alive
p = data_train.groupby("Title")["Survived"].agg(lambda x: 1. * x.sum()/len(x)).plot(kind="bar")
p.set_ylim([0,1])
p


# In[ ]:

# this one i should change a bit
data_train['PclassCat'] = data_train['Pclass'].astype('category')
ax = data_train.loc[data_train['Survived'] == 1].plot(kind='scatter', x='Age', y='Pclass', color='DarkBlue', label='Survived')
data_train.loc[data_train['Survived'] == 0].plot(kind='scatter', x='Age', y='Pclass', color='DarkGreen', label='Died', ax=ax)


# In[ ]:

miss = len(data_train.index) - data_train.count()
h = len(miss > 0)
print
if h > 0:
    print "Missing data train set:"
    print miss[miss > 0]
else:
    print "No missing data in train set"


# In[ ]:

miss = len(data_test.index) - data_test.count()
h = len(miss > 0)
print
if h > 0:
    print "Missing data test set:"
    print miss[miss > 0]
else:
    print "No missing data in test set"


# In[ ]:

# add IsCabin feature
data_train['IsCabin'] = 0
data_train.loc[data_train.Cabin.notnull(), 'IsCabin'] = 1

data_test['IsCabin'] = 0
data_test.loc[data_test.Cabin.notnull(), 'IsCabin'] = 1


# In[ ]:

print len(data_train[data_train.Fare.isnull()])
print len(data_test[data_test.Fare.isnull()])


# In[ ]:

# find correlation between survived and fare + woman or man
print data_train[['Fare', 'Survived']].corr()


# In[ ]:

yo = data_train.groupby(['Title', 'Pclass', 'Survived']).size().unstack()
yo.fillna(0, inplace=True)
print yo


# In[146]:

# working on last name
data_all['LastName'] = data_all.Name.str.extract("(.+),.+")
data_train['LastName'] = data_train.Name.str.extract("(.+),.+")
data_test['LastName'] = data_train.Name.str.extract("(.+),.+")

# find last names that survived

#from sets import Set 
s = data_train[data_train.Survived == 1].LastName.unique()
survived_lastnames = set(s)

sz = data_train.groupby('LastName', group_keys='lol').size()
for (index, value) in sz.iteritems():
    if value == 1: survived_lastnames.discard(value)

# should be 3 variants 
cond = lambda x: 1 if x in survived_lastnames else 0
data_train['RelativeSurvived'] = data_train['LastName'].apply(cond)
data_test['RelativeSurvived'] = data_test['LastName'].apply(cond)

print "Do we have null values somewhere in LastName"
print data_all.LastName.isnull().any()


# In[ ]:

print data_all.groupby(['LastName', 'Pclass', 'Fare']).size()


# In[ ]:



