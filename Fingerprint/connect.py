import KNN
import numpy as np
import pickle
import verification as FPverification

from os import listdir
from os.path import isfile, join

users = KNN.identificate_all()

DATASET_Keystroke = 'DSL-StrongPasswordData.csv'

data, y,data_imposter,y_imposter=KNN.load_data(DATASET_Keystroke, ['subject','sessionIndex', 'rep'])
#print(type(data['total']))
#data['total'].insert(0, 'subject', y)

X_test,Y_test = KNN.split_data(data, y)
#print(X_test)

with open('model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

sample = X_test.iloc[0]
#print(sample)
#exit(0)
sample = sample.to_frame().T


user = KNN.identify_user(knn_model, sample)[0][0]

if user == "Unknown":
    print("User not recognized")
else:
    
    user = str(int(user[1:]))
    #print(user)

    FP_data = "./all_subjects_descriptors.json"
    mypath = "./SOCOFing/Altered/Altered-Easy/"
    pathuser=  user +"_"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f  and f.startswith(pathuser)]
    fingerprint= mypath + onlyfiles[0]

    #print(fingerprint)

    result = FPverification.FPmatching(FP_data, fingerprint, user)["Right_thumb_finger"]
    print(result)
    if result > 3:
        print("Fingerprint recognized")
    else:
        print("Fingerprint not recognized")

    



