from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "confusion_matrix": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds)
    }



#to be pushed to the repo, this file will contain functions to evaluate the performance of the trained model. It will include metrics such as confusion matrix and classification report.