import os
import cv2
import json
from collections import defaultdict
import numpy as np

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

def save_descriptors_to_json(subject_data, output_path):
    """
    Salva i descrittori in un file JSON unico.

    :param subject_data: Dizionario con i descrittori organizzati per soggetto e tipo di dito
    :param output_path: Path del file JSON
    """
    with open(output_path, 'w') as json_file:
        json.dump(subject_data, json_file, indent=4)

def process_images_in_directory(directory_path):
    """
    Elabora tutte le immagini in una directory specificata, estrae i descrittori KAZE
    e li salva in un singolo file JSON con struttura specifica.

    :param directory_path: Path della directory con le immagini
    """
    subject_data = defaultdict(lambda: {"sesso": None})

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.bmp'):
            parts = filename.split('__')
            subject_id = parts[0]
            sex_finger = parts[1].split('.')[0]  # Sesso_tipo_di_dito
            sex, finger_type = sex_finger.split('_', 1)

            image_path = os.path.join(directory_path, filename)
            print(f"Elaborazione di: {image_path}")
            if int(subject_id)>56:
                print("Skipping user: ",subject_id)
                continue
            if finger_type != "Right_thumb_finger":
                print("Skipping finger: ",finger_type," of user: ",subject_id)
                continue
            try:
                descriptors = extract_kaze_descriptors(image_path)
                if descriptors is not None:
                    descriptors_list = descriptors.tolist()  # Convert to list for JSON serialization
                    if subject_data[subject_id]["sesso"] is None:
                        subject_data[subject_id]["sesso"] = sex
                    if finger_type not in subject_data[subject_id]:
                        subject_data[subject_id][finger_type] = []
                    subject_data[subject_id][finger_type].extend(descriptors_list)
            except ValueError as e:
                print(e)

    # Salva tutti i dati in un unico file JSON
    output_path = os.path.join(directory_path, "all_subjects_descriptors.json")
    save_descriptors_to_json(subject_data, output_path)
    print(f"Descrittori salvati per tutti i soggetti in {output_path}")

if __name__ == "__main__":
    path = "./archive/SOCOFing/Real"#input("Inserisci il path della directory contenente le immagini: ")
    process_images_in_directory(path)
