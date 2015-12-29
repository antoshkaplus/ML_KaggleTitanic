
# coding: utf-8

# In[63]:

get_ipython().magic(u'run init.py')


# In[64]:

data_train.columns


# In[65]:

# extracting title
data_train["Title"] = data_train.Name.str.extract(".*, ([a-zA-Z ]*)\. .*")
data_test["Title"] = data_test.Name.str.extract(".*, ([a-zA-Z ]*)\. .*")

print
print "Titles:"
print data_train.Title.value_counts()


# In[66]:

def add_sex_cat(data):
    # all are man
    data["SexCat"] = 0
    data.ix[data.Sex == "female", "SexCat"] = 1
    
    
add_sex_cat(data_train)
add_sex_cat(data_test)


# In[67]:

def add_embarked_cat(data):
    data["EmbarkedCat"] = 0
    data.ix[data.Embarked == "C", "EmbarkedCat"] = 1
    data.ix[data.Embarked == "Q", "EmbarkedCat"] = 2

add_embarked_cat(data_train)
add_embarked_cat(data_test)


# In[68]:

data_test['PclassCat'] = data_test['Pclass'].astype('category')


# In[69]:

data_train.head()


# In[103]:

from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RandForest 


print metrics.SCORERS.keys()


# In[107]:

cols = ["Pclass", "SexCat", "Age", "Fare", "EmbarkedCat", "SibSp"]

train = np.matrix(data_train[cols], dtype=np.float64)
test = np.matrix(data_test[cols], dtype=np.float64)
"""
print np.count_nonzero(test == np.nan)
print type(np.array(np.asarray(test).reshape(-1)))
hh = np.array(np.asarray(test).reshape(-1))
print hh.dtype 
hhh = np.logical_not( np.isfinite(np.asarray(test).reshape(-1)) )
print hh[hhh]
"""
enc = OneHotEncoder(categorical_features=[4])
enc.fit(train)
train = enc.transform(train)
test = enc.transform(test)

solver = LogRegCV(n_jobs=-1)
solver.fit(train, data_train.Survived)
res = solver.predict(test)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_0.csv", index=False)
print solver.score(train, data_train.Survived)

solver = LogRegCV(n_jobs=-1, scoring='roc_auc')
solver.fit(train, data_train.Survived)
res = solver.predict(test)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_1.csv", index=False)
print solver.score(train, data_train.Survived)


solver = LogRegCV(n_jobs=-1, scoring='average_precision')
solver.fit(train, data_train.Survived)
res = solver.predict(test)

res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_2.csv", index=False)
print solver.score(train, data_train.Survived)

"""
solver = LogRegCV(n_jobs=-1)
solver.fit(train, data_train.Survived)
model = SelectFromModel(solver, prefit=True)
print model.get_support()


t_new = model.transform(train)
print t_new.shape

res = solver.predict(test)
res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
res.to_csv("../output/logic_1.csv", index=False)
print solver.score(train, data_train.Survived)

"""

for n in range(1,10):
    # criterion: gini (Gini impurity), entropy
    solver = RandForest(n_estimators=n)
    solver.fit(train, data_train.Survived)
    res = solver.predict(test)
    res = pd.DataFrame({"PassengerId": data_test.PassengerId, "Survived": res})
    res.to_csv("../output/logic_4_%d.csv" % n, index=False)
    print solver.score(train, data_train.Survived)



# In[90]:

metrics.SCORERS.keys()


# In[ ]:




# In[ ]:



