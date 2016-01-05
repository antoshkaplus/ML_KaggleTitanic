import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold as Fold
from sklearn.preprocessing import OneHotEncoder


ALL_COLS = [
    "TitleCat",      # 5
    "Pclass",        # 3
    "SexCat",        # 1
    'AgeCat',        # 3
    "EmbarkedCat",   # 3
    "SibSp",         # 1
    "Parch",         # 1
    'Family',        # 1
    'IsFamily',      # 1
    'IsCabin',       # 1
    'FarePerPerson', # 1
    'Companions',    # 1
    'Pclass_Fare'  # 1  better understanding of when fare is high for particular class
    #'TicketTypeFirstLetter']
    # different work with cabin
]

def get_cv(y):
    return Fold(y, n_folds=10, shuffle=True, random_state=1)

def read_train():
    return pd.read_csv("./../data/train.csv")

def read_test():
    return pd.read_csv("./../data/test.csv")

def read_all():
    tr = read_train()
    tt = read_test()
    return pd.concat([tr, tt], ignore_index=True)

def write(data, filename):
    data[['PassengerId', 'Survived']].to_csv(("../output/%s.csv" % filename), index=False)

def add_int_category(data, col):
    enc = LabelEncoder()
    enc.fit(data[col])
    data[col + "Cat"] = enc.transform(data[col])
    
    
def munge(data):
    add_int_category(data, 'Sex')
    data['IsCabin'] = 1
    data.ix[data.Cabin.isnull(), 'IsCabin'] = 0
    
    #############
    # family
    data['Family'] = data.SibSp + data.Parch
    data['IsFamily'] = 0
    data.ix[data.Family > 0, 'IsFamily'] = 1
    
    ##############
    # lastname extraction
    pattern = "(.+),.+"
    data['LastName'] = data.Name.str.extract(pattern)
    
    #########################
    # title extraction
    pattern = ".*, ([a-zA-Z ]*)\. .*"
    data["Title"] = data.Name.str.extract(pattern)
    similar = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess"]
    data.Title.replace(similar, "Aristocratic", inplace=True)
    data.Title.replace("Mlle", "Miss", inplace=True)
    data.Title.replace("Mme", "Mrs", inplace=True)
    # no husband or relatives found for those ladyes
    data.Title.replace("Ms", "Miss", inplace=True)
    # have to create integer representation
    add_int_category(data, 'Title')
    
    #################
    # predict ages    
    fillna_with_mean = lambda x: x.fillna(x.mean())
    data.Age = data.groupby(['Title', 'Pclass']).Age.transform(fillna_with_mean)

    ########################
    # age categories
    # < 16, 16 - 35, > 35
    data['AgeCat'] = 2
    data.ix[data.Age < 16, 'AgeCat'] = 1
    data.ix[data.Age > 35, 'AgeCat'] = 3
    
    #######################
    # work with fare
    per_person = lambda x: x/len(x)
    data['FarePerPerson'] = data.groupby(['Ticket', 'Fare']).Fare.transform(per_person)
    data.ix[(data.Fare == 19.9667) & (data.LastName == 'Hagland'), 'FarePerPerson'] = 19.9667/2
    
    #######################
    # deal with 0 fare or na
    data.ix[data.FarePerPerson == 0, 'FarePerPerson'] = None
    data.FarePerPerson = data.groupby('Pclass').FarePerPerson.transform(lambda x: x.fillna(x.mean()))
    
    data['Pclass_Fare'] = data.FarePerPerson * data.Pclass
    
    #######################
    # companions
    data['Companions'] = data.groupby('Ticket').Ticket.transform(lambda x: len(x))
    
    #######################
    # embarked fill na and make cat
    data.Embarked.fillna('S', inplace=True)
    add_int_category(data, 'Embarked')
    
def split(data):
    s = data.Survived.isnull()
    train = data[~s]
    test = data[s]
    return train, test

def get_feature_indices(data, cols=ALL_COLS):
    mat = np.matrix(data[cols], dtype=np.float64)
    category_features = []
    # cabin should not be category as it's more like numeric
    for i, c in enumerate(cols):
        if (c == 'Pclass' or c == 'EmbarkedCat' or c == 'TitleCat' 
            or c == 'AgeCat'):
            category_features.append(i)
    enc = OneHotEncoder(categorical_features=category_features)
    enc.fit(mat)
    return enc.feature_indices_
    
# returns train_X, train_Y, test_X
def prepare_matrix(data, cols=ALL_COLS):
    mat = np.matrix(data[cols], dtype=np.float64)
    category_features = []
    # cabin should not be category as it's more like numeric
    for i, c in enumerate(cols):
        if (c == 'Pclass' or c == 'EmbarkedCat' or c == 'TitleCat' 
            or c == 'AgeCat'):
            category_features.append(i)
    enc = OneHotEncoder(categorical_features=category_features)
    enc.fit(mat)
    return enc.fit_transform(mat)
    
    
    
    
    