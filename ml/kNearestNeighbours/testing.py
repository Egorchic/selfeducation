from loo import LeaveOneOut
from kNN import kNearestNeighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=800,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

colors = ['blue', 'red', 'green', 'purple']
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

plt.figure(figsize=(14, 5))

# Обучающая выборка
plt.subplot(1, 2, 1)
for i in range(4):
    mask_train = (y_train == i)
    plt.scatter(X_train[mask_train, 0], X_train[mask_train, 1],
                c=colors[i], label=class_names[i], alpha=0.7, edgecolor='k')
plt.title(f'Training set ({len(X_train)} samples)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Тестовая выборка
plt.subplot(1, 2, 2)
for i in range(4):
    mask_test = (y_test == i)
    plt.scatter(X_test[mask_test, 0], X_test[mask_test, 1],
                c=colors[i], label=class_names[i], alpha=0.7, edgecolor='k')
plt.title(f'Test set ({len(X_test)} samples)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

loo = LeaveOneOut()
loo.fit(X_train, y_train)
best_k = loo.findk()

model_1 = kNearestNeighbors(k=best_k)
model_2 = kNearestNeighbors(k=best_k, weights='distance')

model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)

print(model_1.classification_report(X_test, y_test))
print(model_2.classification_report(X_test, y_test))



