import cv2
import numpy as np
import json


def extract_kaze_descriptors(image_path):
    """
    Estrai descrittori KAZE da un'immagine specificata.

    :param image_path: Path dell'immagine
    :return: Lista di descrittori KAZE
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossibile aprire l'immagine: {image_path}")

    kaze = cv2.KAZE_create(upright=True)
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    if descriptors is not None and len(keypoints) > 10:
        keypoints, descriptors = zip(*sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)[:20])

    return np.array(descriptors)


def match_descriptors(descriptors1, descriptors2, ratio_threshold=0.4):
    """
    Confronta due set di descrittori utilizzando il Brute-Force Matcher.

    :param descriptors1: Primo set di descrittori
    :param descriptors2: Secondo set di descrittori
    :param ratio_threshold: Soglia di ratio per il test di Lowe
    :return: Lista di corrispondenze valide
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Applica il test di Lowe per filtrare i match
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches

def FPmatching(FP_database, fingerprint, id):

    FP_descriptors = extract_kaze_descriptors(fingerprint)

    with open(FP_database, 'r') as json_file:
        data = json.load(json_file)

    if id not in data:
        print(f"DEBUG: L'ID {id} non è presente nel file JSON.")
        return {0}

    matching_scores = {}
    for finger_type, descriptors in data[id].items():
        if finger_type == "sesso":
            continue  # Salta il campo "sesso"
        saved_descriptors = np.array(descriptors, dtype=np.float32)
        matches = match_descriptors(FP_descriptors, saved_descriptors)
        matching_scores[finger_type] = len(matches)

    return matching_scores
    


if __name__ == '__main__':
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