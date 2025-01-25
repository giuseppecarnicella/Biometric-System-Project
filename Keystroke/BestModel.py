import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc



data = pd.read_csv("DSL-StrongPasswordData.csv")


# 51 total
subjects = data["subject"].unique()
# METRICHE: FAR (V),FRR (V),TAR (V),TRR (V),EER (V);ROC (V),Zero error rate,Zero FRR, Zero FAR, Detection error trade off


def evaluateEER_with_FAR_FRR(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    scores = user_scores + imposter_scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    missrates = 1 - tpr  # False Rejection Rate (FRR)
    farates = fpr        # False Acceptance Rate (FAR)
    
    # Find the Equal Error Rate (EER)
    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])  
    eer = x[0] + a * (y[0] - x[0])
    
    # Trova i valori di FAR e FRR al punto EER
    far_at_eer = farates[idx1]
    frr_at_eer = missrates[idx1]
    
    return eer, far_at_eer, frr_at_eer

def plot_figure(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    scores = user_scores + imposter_scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return


def evaluateFAR_FRR(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    scores = user_scores + imposter_scores
    farates, tpr, thresholds = roc_curve(labels, scores)
    
    frrates = 1 - tpr  # False Rejection Rate (FRR)
    
    return list(map(tuple,np.dstack((farates, frrates, thresholds)).reshape(-1,3)))
    


def closest_z_to_target(subjects, target_z):
    """
    For each subject, find the tuple with the closest z to the target value.

    Args:
        subjects (list): A list of subjects, where each subject is a list of tuples (x, y, z).
        target_z (int): The target z value.

    Returns:
        list: A list containing the closest tuple for each subject.
    """
    closest_tuples = []

    for subject in subjects:
        # Find the tuple with the smallest difference in z
        closest_tuple = min(subject, key=lambda t: abs(t[2] - target_z))
        closest_tuples.append(closest_tuple)

    return closest_tuples


def calculate_treshold_scores(new,min=30,max=60,decimal=False):
    
    scores =[]
    if decimal:
        for i in np.linspace(min,max,30):
            values = closest_z_to_target(new,i)
            x = np.mean([val[0] for val in values])
            y = np.mean([val[1] for val in values])
            z = np.mean([val[2] for val in values])
            scores.append((round(x,6),round(y,6),round(i,6),round(z,6)))
        return scores
    else:
        for i in range(min,max):
            values = closest_z_to_target(new,i)
            x = np.mean([val[0] for val in values])
            y = np.mean([val[1] for val in values])
            z = np.mean([val[2] for val in values])
            scores.append((x,y,i,z))
        return scores
    





class ManhattanScaledDetector:
# TRESHOLD  (0.141471, 0.088235, 43.37931, 44.23806)

    def __init__(self, subjects):

        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.mean_vector = self.train.mean().values
        self.mad_vector = self.train.sub(self.mean_vector, axis=1).abs().mean().values

        
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_genuine.iloc[i].values[j] - \
                                self.mean_vector[j]) / self.mad_vector[j]
            self.user_scores.append(cur_score)
            
        for i in range(self.test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_imposter.iloc[i].values[j] - \
                                self.mean_vector[j]) / self.mad_vector[j]
            self.imposter_scores.append(cur_score)

    def evaluate(self):
        eers = []
        fars = []
        frrs = []

        evaluation=[]
        
        for subject in subjects:
            self.user_scores = []
            self.imposter_scores = []

            # Considera il soggetto corrente come genuino e gli altri come impostori
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]

            # Dati di addestramento e test
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]
            
            self.training()
            self.testing()

            # Calcola EER, FAR, FRR
            eer, far, frr = evaluateEER_with_FAR_FRR(self.user_scores, self.imposter_scores)
            
            val= evaluateFAR_FRR(self.user_scores, self.imposter_scores)
            evaluation.append(val)

            eers.append(eer)
            fars.append(far)
            frrs.append(frr)
        
        plot_figure(self.user_scores, self.imposter_scores)
        scores = calculate_treshold_scores(evaluation,40,47,True)


        return { "AVG EER": np.mean(eers),"STDEV EER": np.std(eers),"FAR": np.mean(fars),"FRR": np.mean(frrs),"TRR": 1-np.mean(fars),"TAR": 1-np.mean(frrs)} 


print(ManhattanScaledDetector(subjects).evaluate())