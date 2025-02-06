import tkinter as tk
from tkinter import messagebox
import KNN
import numpy as np
import pickle
import verification as FPverification
from os import listdir
from os.path import isfile, join

# Funzione per avviare l'identificazione
def identify_user():
    sampleID = entry1.get()
    fingerprintID = entry2.get()
    
    if not sampleID or not fingerprintID:
        messagebox.showerror("Errore", "Inserisci entrambi i campi!")
        return
    
    if not sampleID.isdigit() or not fingerprintID.isdigit():
        messagebox.showerror("Errore", "Inserisci valori numerici!")
        return
    
    if int(sampleID) < 1 or int(fingerprintID) < 1:
        messagebox.showerror("Errore", "Inserisci valori positivi!")
        return
    
    if int(sampleID) > 52 or int(fingerprintID) > 52:
        messagebox.showerror("Errore", "Inserisci valori minori di 52!")
        return

    # Caricamento del modello KNN
    with open('model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    # Caricamento dei dati
    DATASET_Keystroke = 'DSL-StrongPasswordData.csv'
    data, y, data_imposter, y_imposter = KNN.load_data(DATASET_Keystroke, ['subject', 'sessionIndex', 'rep'])

    data['total'].insert(0, 'subject', y)
    X_test, Y_test = KNN.split_data(data, y)

    # Seleziona un campione casuale per il test
    subject = 's' + sampleID.zfill(3)
    samples = X_test[X_test['subject'] == subject]
    if samples.empty:
        messagebox.showinfo("Error", "User not present in the test set. Try with another sample ID")
        return
    samples = samples.drop(columns=['subject'])
    sample = samples.sample(n=1)
    #sample = sample.to_frame().T

    user = KNN.identify_user(knn_model, sample)[0][0]

    if user == "Unknown":
        messagebox.showinfo("Outsider detected!", "User not enrolled in the system")
    else:
        user = str(int(user[1:]))
        
        FP_data = "./all_subjects_descriptors.json"
        mypath = "./SOCOFing/Altered/Altered-Easy/"
        pathuser = fingerprintID + "_"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f and f.startswith(pathuser)]
        fingerprint = mypath + onlyfiles[0]

        result = FPverification.FPmatching(FP_data, fingerprint, user)["Right_thumb_finger"]

        if result > 3:
            messagebox.showinfo("User verified!", "Fingerprint recognized")
        else:
            messagebox.showinfo("User not verified!", "Fingerprint not recognized")

# Creazione della finestra principale
root = tk.Tk()
root.title("User Identification")

# Etichette e campi di input
tk.Label(root, text="User keystroke sample ID:").grid(row=0, column=0, padx=10, pady=10)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Fingerprint ID").grid(row=1, column=0, padx=10, pady=10)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=10, pady=10)

# Pulsante per l'identificazione
button = tk.Button(root, text="Identify", command=identify_user)
button.grid(row=2, column=0, columnspan=2, pady=20)

# Avvia la finestra
root.mainloop()
