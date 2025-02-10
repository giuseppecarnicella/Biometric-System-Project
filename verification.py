import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


def extract_kaze_descriptors(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to open the image : {image_path}")

    kaze = cv2.KAZE_create(upright=True)
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    # Selecting only the best 20 descriptors
    if descriptors is not None and len(keypoints) > 10:
        keypoints, descriptors = zip(*sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)[:20])

    return np.array(descriptors)



# Comparing the descriptors of the two images using Brute Force Matcher
def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.6):
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply the Lowe Test to filter the matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)    
    
    return good_matches


# Compute the matching between the external image descriptors and the ones saved in the database
def FPmatching(FP_database, fingerprint, id):

    FP_descriptors = extract_kaze_descriptors(fingerprint)

    with open(FP_database, 'r') as json_file:
        data = json.load(json_file)
    
    if id not in data:
        #print(f"DEBUG: ID {id} not in the JSON file.")
        return {0}

    matching_scores = {}
    for finger_type, descriptors in data[id].items():
        
        if finger_type == "sesso":
            continue 
        saved_descriptors = np.array(descriptors, dtype=np.float32)
        
        matches = match_descriptors(FP_descriptors, saved_descriptors)
        matching_scores[finger_type] = len(matches)
    
    return matching_scores




def test():

    mypathhard = "./SOCOFing/Altered/Altered-Hard/"
    mypathmedium = "./SOCOFing/Altered/Altered-Medium/"
    mypatheasy = "./SOCOFing/Altered/Altered-Easy/"
    listpath = [mypathhard, mypathmedium, mypatheasy]
    FP_data = "./all_subjects_descriptors.json"
    
    
    for mypath in listpath:
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f]
        medium_scores = []
        
        for file in onlyfiles:
            fingerprint = mypath + file
            id = file.split("_")[0]
            if int(id) >46: 
                continue
            matching_results = FPmatching(FP_data, fingerprint, id)

            if matching_results:
                for finger_type, score in matching_results.items():
                    medium_scores.append(score)
                            
            else:
                print("Unable to compute the matching.")

        total_gen = len(medium_scores)
        genuine = sum(1 for score in medium_scores if score >= 3)
        print('GAR '+mypath+' ',genuine/total_gen)
        print('FRR '+mypath+' ',1-genuine/total_gen)


        
        
        imposter_scores = []
        for file in onlyfiles:
            fingerprint = mypath + file
            id = file.split("_")[0]

            # Imposters are in the range [46,52) 
            if int(id) <=46 or int(id)>52 :
                continue

            for id in range(1,47):
                matching_results = FPmatching(FP_data, fingerprint, str(id))
                
                if matching_results:
                    for finger_type, score in matching_results.items():
                        imposter_scores.append(score)
                    
                else:
                    print("Unable to compute the matching.")
        
        total_imp = len(imposter_scores)
        imposters = sum(1 for score in imposter_scores if score >= 3)
        
        print('FAR '+mypath+' ',imposters/total_imp)
        print('GRR '+mypath+' ',1-imposters/total_imp)
        print('\n')


def compute_roc_curve_altered():
    paths = {
        "Altered-Easy": "./SOCOFing/Altered/Altered-Easy/",
        "Altered-Medium": "./SOCOFing/Altered/Altered-Medium/",
        "Altered-Hard": "./SOCOFing/Altered/Altered-Hard/"
    }
    
    FP_data = "./all_subjects_descriptors.json"
    thresholds = range(0,10)
    
    plt.figure(figsize=(8, 6))

    for label, mypath in paths.items():
        far_values = []
        gar_values = []

        for threshold in thresholds:
            genuine_scores = []
            imposter_scores = []

            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f]

            # **Step 1: Compute Genuine Matches**
            for file in onlyfiles:
                fingerprint = mypath + file
                id = file.split("_")[0]
                if int(id) > 46:
                    continue
                matching_results = FPmatching(FP_data, fingerprint, id)
                if matching_results:
                    genuine_scores.extend(matching_results.values())

            # **Step 2: Compute Imposter Matches**
            for file in onlyfiles:
                fingerprint = mypath + file
                id = file.split("_")[0]
                if int(id) <= 46 or int(id) > 52: 
                    continue
                for impostor_id in range(1, 47):
                    matching_results = FPmatching(FP_data, fingerprint, str(impostor_id))
                    if matching_results:
                        imposter_scores.extend(matching_results.values())

            total_genuine = len(genuine_scores)
            total_imposter = len(imposter_scores)

            genuine_accepted = sum(1 for score in genuine_scores if score >= threshold)
            imposters_accepted = sum(1 for score in imposter_scores if score >= threshold)

            gar = genuine_accepted / total_genuine if total_genuine > 0 else 0
            far = imposters_accepted / total_imposter if total_imposter > 0 else 0

            gar_values.append(gar)
            far_values.append(far)

        plt.plot(far_values, gar_values, marker='o', linestyle='-', label=f"ROC {label}")

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("Genuine Acceptance Rate (GAR)")
    plt.title("ROC curve for the altered fingerprints")
    plt.legend()
    plt.grid()
    plt.show()


