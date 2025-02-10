# Biometric System Project

A Lightweight Identification System with Keystroke Dynamics and Fingerprint Verification.

Project made for the Biometrics course @ Sapienza University.

# What it contains

- KNN.py contains all the methods to train and use the KNN model.
- verification.py contains all the methods to use the fingerprint verification model.
- gallery_features.py contains all the methods to create the fingerprint gallery.
- GUI.py contain the demo.

# How to use

- python ./GUI.py to runs the demo.
- python ./final_metrics.py to run the test on the whole system and to get the overall perfomances of it.

- KNN.py has a commented section on the buttom. Uncomment it and run python ./KNN.py to test only the KNN section.


# How the demo works

The demo will ask for three values:
- Sample ID: The ID of the subject from whom a random keystroke sample will be taken.
- Fingerprint ID: The ID of the subject from whom a random fingerprint sample will be taken.
- Difficulty: the level of distortion of the fingerprint
