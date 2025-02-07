import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


from os import listdir
from os.path import isfile, join


# Estrai descrittori KAZE da un'immagine specificata.
def extract_kaze_descriptors(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossibile aprire l'immagine: {image_path}")

    kaze = cv2.KAZE_create(upright=True)
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    if descriptors is not None and len(keypoints) > 10:
        keypoints, descriptors = zip(*sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)[:20])

    return np.array(descriptors)



# Confronta due set di descrittori utilizzando il Brute-Force Matcher.
def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.6):
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Applica il test di Lowe per filtrare i match
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    #print(good_matches)
    
    
    return good_matches


# Esegue il matching tra i descrittori dell'immagine esterna e quelli salvati nel database.
def FPmatching(FP_database, fingerprint, id):

    FP_descriptors = extract_kaze_descriptors(fingerprint)
    

    with open(FP_database, 'r') as json_file:
        data = json.load(json_file)

    if id not in data:
        print(f"DEBUG: L'ID {id} non è presente nel file JSON.")
        return {0}

    matching_scores = {}
    #print(data[id].items())
    for finger_type, descriptors in data[id].items():
        
        if finger_type == "sesso":
            continue  # Salta il campo "sesso"
        saved_descriptors = np.array(descriptors, dtype=np.float32)
        
        matches = match_descriptors(FP_descriptors, saved_descriptors)
        matching_scores[finger_type] = len(matches)
        shape = {k: v for k, v in matching_scores.items()}
        #print(shape)
    
    return matching_scores




def test():

    #mypathreal = "./SOCOFing/Real/"
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
            if int(id) >46: # METTI 46
                continue
            matching_results = FPmatching(FP_data, fingerprint, id)
            #print(matching_results)
            #print(id)
            if matching_results:
                #print(f"Risultati del matching per l'ID dichiarato {id}:")
                for finger_type, score in matching_results.items():
                    #print(f"- {finger_type}: {score} corrispondenze valide")
                    medium_scores.append(score)
                            
            else:
                print("Impossibile effettuare il matching.")

        #print(medium_scores)
        total_gen = len(medium_scores)
        genuine = sum(1 for score in medium_scores if score >= 3)
        print('GAR '+mypath+' ',genuine/total_gen)
        print('FRR '+mypath+' ',1-genuine/total_gen)


        
        
        imposter_scores = []
        for file in onlyfiles:
            fingerprint = mypath + file
            id = file.split("_")[0]
            if int(id) <=46 or int(id)>52 :
                continue

            for id in range(1,47):
                matching_results = FPmatching(FP_data, fingerprint, str(id))
                
                if matching_results:
                    #print(f"Risultati del matching per l'ID dichiarato {id}:")
                    for finger_type, score in matching_results.items():
                        #print(f"- {finger_type}: {score} corrispondenze valide")
                        imposter_scores.append(score)
                    
                else:
                    print("Impossibile effettuare il matching.")
        
        total_imp = len(imposter_scores)
        imposters = sum(1 for score in imposter_scores if score >= 3)
        #print(imposter_scores)
        
        print('FAR '+mypath+' ',imposters/total_imp)
        print('GRR '+mypath+' ',1-imposters/total_imp)
        print('\n')



import matplotlib.pyplot as plt
import numpy as np

def compute_roc_curve_altered():
    paths = {
        "Altered-Easy": "./SOCOFing/Altered/Altered-Easy/",
        "Altered-Medium": "./SOCOFing/Altered/Altered-Medium/",
        "Altered-Hard": "./SOCOFing/Altered/Altered-Hard/"
    }
    
    FP_data = "./all_subjects_descriptors.json"
    thresholds = range(0,10)  # Soglie da 1 a 20
    
    plt.figure(figsize=(8, 6))

    for label, mypath in paths.items():
        far_values = []
        gar_values = []

        for threshold in thresholds:
            genuine_scores = []
            imposter_scores = []

            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f]

            # **Step 1: Calcolare Genuine Matches**
            for file in onlyfiles:
                fingerprint = mypath + file
                id = file.split("_")[0]
                if int(id) > 46:  # Utenti registrati
                    continue
                matching_results = FPmatching(FP_data, fingerprint, id)
                if matching_results:
                    genuine_scores.extend(matching_results.values())

            # **Step 2: Calcolare Imposter Matches**
            for file in onlyfiles:
                fingerprint = mypath + file
                id = file.split("_")[0]
                if int(id) <= 46 or int(id) > 52:  # ID 47-52 sono impostori
                    continue
                for impostor_id in range(1, 47):  # Confronto con utenti reali
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

        # Plotta la curva ROC per questa categoria
        plt.plot(far_values, gar_values, marker='o', linestyle='-', label=f"ROC {label}")

    # Configurazione del grafico
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("Genuine Acceptance Rate (GAR)")
    plt.title("Curva ROC per le impronte alterate")
    plt.legend()
    plt.grid()
    plt.show()




if __name__ == '__main__':
    '''
    FP_data = "./all_subjects_descriptors.json"
    
    fingerprint = "./archive/SOCOFing/Altered/Altered-Hard/1__M_Right_thumb_finger_CR.BMP"
        #"./archive/SOCOFing/Real/1__M_Right_thumb_finger.BMP"   # Inserisci il path dell'immagine esterna
    id = "1"  # ID dichiarato dall'immagine esterna
    matching_results = FPmatching(FP_data, fingerprint, id)

    if matching_results:
        print(f"Risultati del matching per l'ID dichiarato {id}:")
        for finger_type, score in matching_results.items():
            print(f"- {finger_type}: {score} corrispondenze valide")
    else:
        print("Impossibile effettuare il matching.")
    '''
