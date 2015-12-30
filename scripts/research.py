
# coding: utf-8

# In[153]:

import init
reload(init)
from init import * 


# In[154]:

data_train.columns


# In[155]:

def add_sex_cat(data):
    # all are man
    data["SexCat"] = 0
    data.ix[data.Sex == "female", "SexCat"] = 1
    
    
add_sex_cat(data_train)
add_sex_cat(data_test)


# In[156]:

def add_embarked_cat(data):
    data["EmbarkedCat"] = 0
    data.ix[data.Embarked == "C", "EmbarkedCat"] = 1
    data.ix[data.Embarked == "Q", "EmbarkedCat"] = 2

add_embarked_cat(data_train)
add_embarked_cat(data_test)


# In[157]:

# lets add IsFamily column
def add_is_family(data):
    data['Family'] = data.SibSp + data.Parch
    data['IsFamily'] = 0
    data.ix[data.SibSp + data.Parch > 0, 'IsFamily'] = 1
    
    
add_is_family(data_train)
add_is_family(data_test)


# In[158]:

data_train.head()


# In[184]:

from sklearn.preprocessing import LabelEncoder


# fill-in cabin letter use '0' if no cabin letter present
def add_cabin_letter(data):
    s = data['Cabin'].notnull()
    data['CabinLetter'] = '0'
    data['CabinNumber'] = '0'
    
    data.ix[s, 'CabinLetter'] = data[s].Cabin.str.extract("([a-zA-Z ]+).*")
    data.ix[data.CabinLetter >= 'F', 'CabinLetter'] = 'F'
    #data.ix[data.CabinLetter >= 'D', 'CabinLetter'] = 'D'
    
    data.ix[s, 'CabinNumber'] = data[s].Cabin.str.extract("[a-zA-Z].*?(\d+)")
    data.CabinNumber.fillna('0', inplace=True)
    data.CabinNumber = data.CabinNumber.astype(int)
    
    enc = LabelEncoder()
    enc.fit(data.CabinLetter)
    # do we really need this???
    data['CabinLetterCat'] = enc.transform(data.CabinLetter) 
    
    # combine number and letter of cabin
    #data['CabinLetNumComb'] = data['CabinLetterCat'] .map('${:,.2f}'.format)
    
    
add_cabin_letter(data_train)
add_cabin_letter(data_test)


#print data_train.groupby(["CabinLetter", "Sex", "Survived"]).size()
#p = data_train.sort_values(by='CabinNumber').groupby(["CabinNumber", "Sex", "Survived"]).size()
#print p

data_all['TicketType'] = data_all.Ticket.str.extract("(.*) \d+")
data_all['TicketType'].fillna('0', inplace=True)

data_train['TicketType'] = data_train.Ticket.str.extract("(.*) \d+")
data_train['TicketType'].fillna('0', inplace=True)

with pd.option_context('display.max_rows', 10000, 'display.max_columns', 3):
    p_0 = data_all.groupby(['TicketType', 'Survived']).size()
    p_1 = data_train.groupby(['TicketType', 'Survived']).size()
    print pd.concat([p_0, p_1], axis=1)

data_train.groupby(["Ticket"]).size()

# lets try to extract last digits
data_train['TicketType'] = data_train.Ticket.str.extract("(.*) \d+")
data_train['TicketType'].fillna('0', inplace=True)

data_train['TicketTypeFirstLetter'] = data_train.TicketType.str[0]

#print data_train.groupby(['TicketTypeFirstLetter', 'Sex', 'Survived']).size()
#df['Date'] = df['Date'].apply(lambda x: int(str(x)[-4:]))

# need fcking graph here

# lets take a look at ticket numbers where there are only ticket numbers... nothing else


yo = data_train.groupby(['CabinLetter', 'Survived']).size()
#yo = yo.unstack()
#yo.fillna(0, inplace=True)
#print yo


# In[160]:

from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RandForest 


print metrics.SCORERS.keys()


# In[161]:

print data_train.TitleCat.max() + 1
print len(data_train.Pclass.value_counts())
print len(data_train.SexCat.value_counts())


# In[162]:

# lets see what we can do with the ticket


# In[163]:

# don't need IsFamily as Family exists
# after adding title logic reg became worth. random tree becomes better
all_cols = ["TitleCat", "Pclass", "SexCat", "Age", "Fare", "EmbarkedCat", 
            "SibSp", "Parch", 'Family', "CabinLetterCat", 'CabinNumber', 'TicketTypeFirstLetter']
# cols : list of columns
def prepare_matrices(cols=all_cols):
    # sex is binary
    mat_train = np.matrix(data_train[cols], dtype=np.float64)
    mat_test = np.matrix(data_test[cols], dtype=np.float64)

    category_features = []
    # cabin should not be category as it's more like numeric
    for i, c in enumerate(cols):
        if (c == 'Pclass' or c == 'EmbarkedCat' or c == 'TitleCat' 
            or c == 'CabinLetterCat' or c == 'TicketTypeFirstLetter'):
            category_features.append(i)
    enc = OneHotEncoder(categorical_features=category_features)
    enc.fit(mat_train)
    return tuple(map(enc.transform, [mat_train, mat_test]))


# In[164]:

data_train[['Survived'] + all_cols].corr()


# In[165]:

# big Pclass, IsCabin;

# we have to do feature selection


# lets find out what important features we have here
from sklearn.feature_selection import SelectKBest, f_classif


tt, mission = prepare_matrices()
print tt.toarray()[0,:]
print mission.toarray()[0,:]
selector = SelectKBest(f_classif, k=5)
selector.fit(tt, data_train.Survived)
scores = -np.log10(selector.pvalues_)

print scores
#plt.xticks(range(len(predictors)), predictors, rotation='vertical')


# In[ ]:



