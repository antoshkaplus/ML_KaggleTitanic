from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv("./../data/train.csv")
data_test = pd.read_csv("./../data/test.csv")

print data_train
print data_test

print
print "Missing data train set:"
miss = len(data_train.index) - data_train.count()
print 1. * miss / len(data_train.index)

print
print "Missing data test set:"
miss = len(data_test.index) - data_test.count()
print 1. * miss / len(data_test.index)

# looking at features distribution

# should create multiple data frames to have multiple plots
survived = pd.DataFrame( data_train.Survived.value_counts() )
survived.index = ["Died", "Survived"]
#survived.plot(kind="bar", rot=0, title="Distribution of survived passagers")

pclass = pd.DataFrame( data_train.Pclass.value_counts() )
#pclass.plot(kind="bar", rot=0, title="Distribution by class")

sex = pd.DataFrame( data_train.Sex.value_counts() )
#sex.plot(kind="bar", rot=0, title="Distribution by sex")

# sibsp: Number of Siblings/Spouses Aboard
# parch: Number of Parents/Children Aboard

# Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 2 NA
s = data_train.Embarked
s.fillna(s.value_counts().idxmax(), inplace=True)
embarked = pd.DataFrame( data_train.Embarked.value_counts() )
#embarked.plot(kind="bar", rot=0, title="Distribution by place of embarkation")


# feature distribution + survival

d = data_train.groupby(["Pclass", "Survived"]).size()
d = d.unstack()
#d.plot(kind="bar", rot=0, stacked=True)

d = data_train.groupby(["Sex", "Survived"]).size()
d = d.unstack()
#d.plot(kind="bar", rot=0, stacked=True)

d = data_train.groupby(["Embarked", "Survived"]).size()
d = d.unstack()
#d.plot(kind="bar", rot=0, stacked=True)


# extracting title
data_train["Title"] = data_train.Name.str.extract(".*, ([a-zA-Z]*)\. .*")
data_test["Title"] = data_test.Name.str.extract(".*, ([a-zA-Z]*)\. .*")

print
print "Titles:"
print data_train.Title.value_counts()

# want to preserve column order
d = OrderedDict()
d["Count"] = len
d["Missing"] = lambda x: x.isnull().sum()
d["Mean"] = np.mean
print
print data_train.groupby("Title")["Age"].agg(d)


# fill age na values with means from Title

def func(x):
    x.fillna(x.mean(), inplace=True)
    #print x
    return x

data_train.Age = data_train.groupby("Title").Age.apply(func)
#print data_train.groupby("Title").apply(lambda x: x.Age.fillna(x.Age.mean(), inplace=True))
print data_train.groupby("Title")["Age"].agg(d)
# should do the same with test data


# fill zero fare


plt.show()
