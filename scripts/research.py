
# coding: utf-8

# In[29]:

import init
reload(init)
from init import * 


# In[30]:

data_train.columns


# In[31]:

def add_sex_cat(data):
    # all are man
    data["SexCat"] = 0
    data.ix[data.Sex == "female", "SexCat"] = 1
    
    
add_sex_cat(data_train)
add_sex_cat(data_test)


# In[32]:

def add_embarked_cat(data):
    data["EmbarkedCat"] = 0
    data.ix[data.Embarked == "C", "EmbarkedCat"] = 1
    data.ix[data.Embarked == "Q", "EmbarkedCat"] = 2

add_embarked_cat(data_train)
add_embarked_cat(data_test)


# In[33]:

# lets add IsFamily column
def add_is_family(data):
    data['Family'] = data.SibSp + data.Parch
    data['IsFamily'] = 0
    data.ix[data.SibSp + data.Parch > 0, 'IsFamily'] = 1
    
    
add_is_family(data_train)
add_is_family(data_test)


# In[34]:

data_train.head()


# In[35]:

from sklearn.preprocessing import LabelEncoder

# fill-in cabin letter use '0' if no cabin letter present
def add_cabin_letter(data):
    s = data['Cabin'].notnull()
    data['CabinLetter'] = '0'
    data.ix[s, 'CabinLetter'] = data[s].Cabin.str.extract("([a-zA-Z ]+).*")
    enc = LabelEncoder()
    enc.fit(data.CabinLetter)
    # do we really need this???
    data['CabinLetterCat'] = enc.transform(data.CabinLetter) 
    
add_cabin_letter(data_train)
add_cabin_letter(data_test)

#yo = data_train.groupby(['CabinLetter', 'Sex', 'Survived']).size()
#yo = yo.unstack()
#yo.fillna(0, inplace=True)
#print yo


# In[36]:

from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RandForest 


print metrics.SCORERS.keys()


# In[37]:

# don't need IsFamily as Family exists
# after adding title logic reg became worth. random tree becomes better
all_cols = ["TitleCat", "Pclass", "SexCat", "Age", "Fare", "EmbarkedCat", 
            "SibSp", "Parch", "IsCabin", 'Family', "CabinLetterCat"]

# cols : list of columns
def prepare_matrices(cols=all_cols):
    # sex is binary
    mat_train = np.matrix(data_train[cols], dtype=np.float64)
    mat_test = np.matrix(data_test[cols], dtype=np.float64)

    category_features = []
    for i, c in enumerate(cols):
        if c == 'Pclass' or c == 'EmbarkedCat' or c == 'Title':
            category_features.append(i)
    enc = OneHotEncoder(categorical_features=category_features)
    enc.fit(mat_train)
    return tuple(map(enc.transform, [mat_train, mat_test]))


# In[38]:

data_train[['Survived'] + all_cols].corr()


# In[39]:

# big Pclass, IsCabin;

# we have to do feature selection

