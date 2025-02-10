import tkinter as tk
from tkinter import messagebox
import KNN
import pandas as pd
import numpy as np
import pickle
import verification as FPverification
from os import listdir
from os.path import isfile, join




def identify_user():
    sampleID = entry1.get()
    fingerprintID = entry2.get()
    difficulty_level = difficulty.get()
    
    if not sampleID or not fingerprintID:
        messagebox.showerror("Error", "One or more fields are empty!")
        return
    
    if not sampleID.isdigit() or not fingerprintID.isdigit():
        messagebox.showerror("Error", "Numerical values only!")
        return
    
    if int(sampleID) < 1 or int(fingerprintID) < 1:
        messagebox.showerror("Error", "Positive values only!")
        return
    
    if int(sampleID) > 52 or int(fingerprintID) > 52:
        messagebox.showerror("Error", "Only values lower than 52!")
        return

    
    with open('model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    
    DATASET_Keystroke = 'DSL-StrongPasswordData.csv'
    data, y, data_imposter, y_imposter = KNN.load_data(DATASET_Keystroke, ['subject', 'sessionIndex', 'rep'])

    # To know to whom the sample belongs we put the subject back
    data['total'].insert(0, 'subject', y)
    data_imposter['total'].insert(0, 'subject', y_imposter)
    X_genuine, _ = KNN.split_data(data, y)
    X_imposter, _ = KNN.split_data(data_imposter, y_imposter)
    X_test = pd.concat([X_genuine, X_imposter])

    # All the subjects are in the format sXXX
    subject = 's' + sampleID.zfill(3)
    samples = X_test[X_test['subject'] == subject]

    if samples.empty:
        messagebox.showinfo("Error", "User not present in the test set. Try with another sample ID lower than 52")
        return
    # Select a random sample
    samples = samples.drop(columns=['subject'])
    sample = samples.sample(n=1)

    user = KNN.identify_user(knn_model, sample)[0][0]
    print(user)
    if user == "Unknown":
        messagebox.showinfo("Outsider detected!", "User not enrolled in the system")
    else:
        user = str(int(user[1:]))
        
        FP_data = "./all_subjects_descriptors.json"
        mypath = "./SOCOFing/Altered/Altered-" + difficulty_level + "/"
        pathuser = fingerprintID + "_"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and "thumb" in f and "Right" in f and f.startswith(pathuser)]
        fingerprint = mypath + onlyfiles[0]

        result = FPverification.FPmatching(FP_data, fingerprint, user)["Right_thumb_finger"]

        if result > 3:
            messagebox.showinfo("User verified!", "Fingerprint recognized")
        else:
            messagebox.showinfo("User not verified!", "Fingerprint not recognized")

root = tk.Tk()
root.title("User Identification")

tk.Label(root, text="User keystroke sample ID:").grid(row=0, column=0, padx=10, pady=10)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Fingerprint ID").grid(row=1, column=0, padx=10, pady=10)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Difficulty Level:").grid(row=2, column=0, padx=10, pady=10)
difficulty = tk.StringVar(value="Easy")  
difficulty_menu = tk.OptionMenu(root, difficulty, "Easy", "Medium", "Hard")
difficulty_menu.grid(row=2, column=1, padx=10, pady=10)
difficulty_menu.configure(width=13)


button = tk.Button(root, text="Identify", command=identify_user)
button.grid(row=3, column=0, columnspan=2, pady=20)

root.mainloop()
