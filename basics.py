import pickle

with open('journalmodel_xgboost.pkl', 'wb') as file:
    pickle.dump(classifier, file)