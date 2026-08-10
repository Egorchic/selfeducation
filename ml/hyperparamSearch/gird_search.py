from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd


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


#Init/train grid
param_grid = {
    'logreg__C': [0.001, 0.01, 1, 10, 100],
    'logreg__class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    verbose=1
)

grid.fit(X_train, y_train)


#Print results
print('Best params:', grid.best_params_)
print('CV ROC-AUC:', grid.best_score_)

y_proba = grid.predict_proba(X_test)[:, 1]

print('Test ROC-AUC:', roc_auc_score(y_test, y_proba))

results = pd.DataFrame(grid.cv_results_)

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



