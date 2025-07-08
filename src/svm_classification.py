import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

def train_svm(X_train, y_train, kernel='poly', gamma=0.5, C=0.1, mi_k=None):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    selector = None
    if mi_k:
        selector = SelectKBest(mutual_info_classif, k=mi_k)
        X_train_scaled = selector.fit_transform(X_train_scaled, y_train)

    clf = SVC(kernel=kernel, gamma=gamma, C=C)
    clf.fit(X_train_scaled, y_train)
    return clf, scaler, selector
