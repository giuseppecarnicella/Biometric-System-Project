import KNN
import pandas as pd
import pickle
import numpy as np
import verification as FPverification
from os import listdir
from os.path import isfile, join
from collections import defaultdict


def metrics():

    with open('model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    DATASET_Keystroke = 'DSL-StrongPasswordData.csv'
    data, y, data_imposter, y_imposter = KNN.load_data(DATASET_Keystroke, ['subject', 'sessionIndex', 'rep'])
    X_genuine, Y_genuine = KNN.split_data(data, y)
    X_imposter, Y_imposter = KNN.split_data(data_imposter, y_imposter)


    user_genuine = KNN.identify_user(knn_model, X_genuine)[0]
    user_imposter = KNN.identify_user(knn_model, X_imposter)[0]
    

    user_genuine = np.column_stack((user_genuine, Y_genuine))
    user_imposter = np.column_stack((user_imposter, Y_imposter))
    FP_data = "./all_subjects_descriptors.json"
    mypath = "./SOCOFing/Altered/Altered-Medium/"
    
    
    frr = 0
    far = 0
    gar = 0
    grr = 0

    genuine = len(X_genuine)
    imposter = len(X_imposter)
    temp = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f ]

    onlyfiles = defaultdict(list)
    

    # Creation of a dictionary for fast retrivial of the fingerprint paths
    for s in temp:
        key = s[:3] 
        if key[2] != "_":
            continue
        # skipping all the users with id >= 70 
        if int(key[0]) > 6 and key[1] != "_":
            continue
        onlyfiles[key].append(s)
    

    i = 0
    for pred,y in user_genuine:        
        i +=1
        if pred == "Unknown":
            frr += 1
            continue
            
        # all the image names ids are 3 digits long with a "_" at the end
        fingerprint = join(mypath, str(onlyfiles[str(int(y[1:])).ljust(3, '_')][0]))
        result = FPverification.FPmatching(FP_data, fingerprint, str(int(pred[1:])))["Right_thumb_finger"]

        if result > 3 and y == pred:
            gar += 1
        elif result <= 3 and y == pred:
            frr += 1
        elif result > 3 and y != pred:
            far += 1
        elif result <= 3 and y != pred:
            frr += 1
        else:
            # should never happen
            print("Error")
        if i % 100 == 0:
            print(f"Progress: {i}/{genuine}")
            print('GAR:',gar/i)
            print('FRR:',frr/i)


    print('GAR:',gar/genuine)
    print('FRR:',frr/genuine)
    i =0
    for pred,y in user_imposter:
        i+=1
        if pred == "Unknown":
            grr += 1
            continue
        
        fingerprint = join(mypath, str(onlyfiles[str(int(y[1:])).ljust(3, '_')][0]))
        result = FPverification.FPmatching(FP_data, fingerprint, str(int(pred[1:])))["Right_thumb_finger"]
        
        if result > 3:
            far +=1
        else:
            grr += 1
        

            
    print('FAR:',far/imposter)
    print('GRR:',grr/imposter)
    

if __name__ == "__main__": 
    metrics()
