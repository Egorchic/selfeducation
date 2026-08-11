import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification


#Data
X, y = make_classification(
    n_samples=3000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


#Permutation importance
imp = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    scoring='accuracy'
)


#Result
result = pd.DataFrame({
    'feature': [f'Z{i}' for i in range(X.shape[1])],
    'importance': imp.importances_mean,
    'std': imp.importances_std
})

print(result.sort_values('importance', ascending=False))




