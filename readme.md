Titanic Kaggle problem solution

we don't divide on training and validation set, because we are interested in
only in relative performance (better or worse).

to find out error we use 10-fold cross validation.


current best relative result:
logic regress: 0.831706957213

after ages cat, fare per person
0.83950617284
0.838383838384  - after adding Pclass * FarePerPerson


using random forest with params
0.833894500561 // with best params

Product of Pclass and FarePerPerson creates overfitness
# overfitted
0.83950617284 // new awesome forest

// best result
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

best public score: 0.80383

should look more into families. cabins and ticket numbers.
find a way to not overfit.
feature engineering should be helpful

maybe remove features that we don't need
