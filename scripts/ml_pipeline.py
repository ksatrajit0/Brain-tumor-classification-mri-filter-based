import pandas as pd
from src.svm_classification import train_svm
from src.evaluate import evaluate_model
from src.config import CSV_TRAIN_PATH, CSV_TEST_PATH, LABELS_DICT

def main(use_mutual_info=True, k_features=mi_k):
    # Load feature CSVs
    train_df = pd.read_csv(CSV_TRAIN_PATH)
    test_df = pd.read_csv(CSV_TEST_PATH)

    # Extract features and labels
    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    # Train SVM (with or without mutual information)
    clf, scaler, selector = train_svm(X_train, y_train, mi_k=k_features if use_mutual_info else None)

    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    if selector:
        X_test_scaled = selector.transform(X_test_scaled)

    y_pred = clf.predict(X_test_scaled)
    class_names = [k for k, v in sorted(LABELS_DICT.items(), key=lambda item: item[1])]
    evaluate_model(y_test, y_pred, class_names)

if __name__ == "__main__":
    main()
