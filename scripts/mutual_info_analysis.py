import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def run_mi_experiment(X_train, y_train, X_test, y_test, k_range):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel='poly', gamma=0.5, C=0.1)
    accuracy_list = []

    for k in k_range:
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train_k = selector.fit_transform(X_train_scaled, y_train)
        X_test_k = selector.transform(X_test_scaled)
        clf.fit(X_train_k, y_train)
        y_pred = clf.predict(X_test_k)
        accuracy_list.append(accuracy_score(y_test, y_pred))

    plt.plot(k_range, accuracy_list, marker='o')
    plt.title('Accuracy vs. Number of Features')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
