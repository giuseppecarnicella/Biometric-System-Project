# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import euclidean, cityblock
import pickle


plt.style.use('ggplot')

# %%
DATASET_2 = 'DSL-StrongPasswordData.csv'
df = pd.read_csv(DATASET_2)
subject = df['subject']
df.head()

# %%
def load_data(file_name, col_to_remove):
    df = pd.read_csv(file_name)

    intruders = 5
    sps = 400
    ending = df.shape[0] - intruders*sps

    df2= df.iloc[ending:]
    df = df.iloc[:ending]

    

    H_columns_genuine  = [col for col in df.columns if col.startswith('H')]
    DD_columns_genuine = [col for col in df.columns if col.startswith('DD')]
    UD_columns_genuine = [col for col in df.columns if col.startswith('UD')]

    data = {}
    data['total'] = df.drop(columns=col_to_remove)
    data['H']     = df[H_columns_genuine]
    data['DD']    = df[DD_columns_genuine]
    data['UD']    = df[UD_columns_genuine]
    data['pca3']  = pd.DataFrame(PCA(n_components=3).fit_transform(data['total']))
   # data['pca10'] = pd.DataFrame(PCA(n_components=10).fit_transform(data['total']))
    

    H_columns_imposter  = [col for col in df.columns if col.startswith('H')]
    DD_columns_imposter = [col for col in df.columns if col.startswith('DD')]
    UD_columns_imposter = [col for col in df.columns if col.startswith('UD')]

    data_imposter = {}
    data_imposter['total'] = df2.drop(columns=col_to_remove)
    data_imposter['H']     = df2[H_columns_imposter]
    data_imposter['DD']    = df2[DD_columns_imposter]
    data_imposter['UD']    = df2[UD_columns_imposter]
    data_imposter['pca3']  = pd.DataFrame(PCA(n_components=3).fit_transform(data_imposter['total']))
    

    
    return data, df['subject'].values, data_imposter, df2['subject'].values


def calculate_KNN(data, y):
    """
    Funzione che addestra il modello KNN su tutti i dati e restituisce il modello addestrato.
    """
    X = data['total']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    
    # Ricerca dei migliori iperparametri per KNN
    n_neighbors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    parameters = dict(n_neighbors=n_neighbors)
    clf = KNeighborsClassifier()
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, Y_train)
    
    # Ottieni il miglior modello KNN
    best_model = grid.best_estimator_

    # Predizioni per valutazione ---> restituisce il soggetto predetto
    Y_pred = grid.predict(X_test)
    

    return best_model, X_test, Y_test



def split_data(data, y):
    """
    Funzione per dividere i dati in training e testing set.
    """
    X = data['total']
    _, X_test, _, Y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return X_test, Y_test


def identify_user(model, samples ,threshold=0.5):
    """
    Identifica il soggetto a partire da un campione dato.
    
    Parameters:
    - model: modello KNN allenato.
    - sample: array con le metriche di keystroke dell'utente (1xN).
    
    Returns:
    - Predizione del soggetto più simile.
    """

    
    subject = model.predict(samples)  # Predice il soggetto
   
    proba = np.max(model.predict_proba(samples),axis=1)  # Probabilità di appartenenza
    subject = np.where(proba <= threshold, 'Unknown', subject) # Se la probabilità è bassa, il soggetto è sconosciuto

    return subject,proba






# %%

def identificate_all():

    data, y,data_imposter,y_imposter = load_data(DATASET_2, ['subject', 'sessionIndex', 'rep'])
    Y = pd.get_dummies(y).values

    #eer2_2 = brentq(lambda x : 1. - x - interp1d(fpr2_2, tpr2_2)(x), 0., 1.)
    # Otteniamo il modello e i dati di testing
    #knn_model, X_test, Y_test = calculate_KMeans(data, y)

    X_test,Y_test = split_data(data, y)

    #load
    with open('model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    return identify_user(knn_model, X_test)






# %% [markdown]
# METRICHE

def metrics(knn_model, data_imposter, X_test):
    X_imp = data_imposter['total']

    gen_acc=0
    imp_acc=0
    gen_total= X_test.shape[0]
    imp_total= X_imp.shape[0]

    genuine_y_pred=identify_user(knn_model, X_test)[0]
    impostor_y_pred=identify_user(knn_model, X_imp)[0]

    gen_acc=np.count_nonzero(np.not_equal(genuine_y_pred, "Unknown"))
    imp_acc=np.count_nonzero(np.not_equal(impostor_y_pred, "Unknown"))


    print(np.not_equal(genuine_y_pred, "Unknown"))
    print('TAR:',gen_acc/gen_total)
    print('FRR:',1-gen_acc/gen_total)
    print('FAR:',imp_acc/imp_total)
    print('TRR:',1-imp_acc/imp_total)
'''
data, y,data_imposter,y_imposter=load_data(DATASET_2, ['subject', 'sessionIndex', 'rep'])
model,X_test,Y_test = calculate_KMeans(data, y)
metrics(model, data_imposter, X_test)
'''

