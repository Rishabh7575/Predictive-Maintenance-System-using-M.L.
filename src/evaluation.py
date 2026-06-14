from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds)
    }

    return results

#Pushed to the repo, this file will contain functions to evaluate the performance of the trained model. It will include metrics such as confusion matrix and classification report.