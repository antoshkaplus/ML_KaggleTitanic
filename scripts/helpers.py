from sklearn.preprocessing import LabelEncoder

def add_int_category(data, col):
    enc = LabelEncoder()
    enc.fit(data[col])
    data[col + "Cat"] = enc.transform(data[col])
    
def munge(data):
    #############
    # family
    data['Family'] = data.SibSp + data.Parch
    
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
    