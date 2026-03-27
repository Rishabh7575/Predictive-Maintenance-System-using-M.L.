# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score

# def train_models(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     lr = LogisticRegression(max_iter=1000)
#     lr.fit(X_train, y_train)
#     lr_f1 = f1_score(y_test, lr.predict(X_test))

#     rf = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1
#     )
#     rf.fit(X_train, y_train)
#     rf_f1 = f1_score(y_test, rf.predict(X_test))

#     return {
#         "logistic_f1": lr_f1,
#         "random_forest_f1": rf_f1
#     }




# #to be pushed to the repo, this file will contain functions to train baseline models such as logistic regression and random forest. It will return the F1 scores for both models.