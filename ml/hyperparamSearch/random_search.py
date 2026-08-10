import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from scipy.stats import uniform


#Load dataset
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#Make pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])


#Init/train searcher
distribution = {
    'logreg__C': uniform(0.01, 9.99),
    'logreg__class_weight': [None, 'balanced']
}

clf = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=distribution,
    n_iter=15,
    scoring='roc_auc',
    n_jobs=-1,
    cv=5,
    verbose=1,
)

clf.fit(X_train, y_train)


#Print results
print('Best params:', clf.best_params_)
print('Best ROC-AUC:', clf.best_score_)

y_proba = clf.predict_proba(X_test)[:, 1]

print('Test ROC-AUC:', roc_auc_score(y_test, y_proba))

results = pd.DataFrame(clf.cv_results_)

print(
    results[
        [
            'param_logreg__C',
            'param_logreg__class_weight',
            'mean_test_score',
            'std_test_score',
            'rank_test_score'
        ]
    ].sort_values('rank_test_score')
)

