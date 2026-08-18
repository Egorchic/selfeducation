from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


#Data
data = fetch_california_housing(as_frame=True)

X, y = data.data, data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
)


def lr_experiments(lr: float):
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=lr,
        max_depth=5,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=False
    )

    eval_res = model.evals_result()

    train_los = eval_res['validation_0']['rmse']
    val_los = eval_res['validation_1']['rmse']


    plt.figure()
    plt.plot(list(range(1500)), train_los, 'b', label='Train RMSE loss')
    plt.plot(list(range(1500)), val_los, 'r', label='Val RMSE loss')

    plt.title(f'Train and val loss comp, lr={lr}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'learning_rate_choosing/losses_comp_{lr}.png')


def n_estimators_experiments(n: int):
    model = XGBRegressor(
        n_estimators=n,
        learning_rate=0.003,
        max_depth=5,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=False
    )

    eval_res = model.evals_result()

    train_los = eval_res['validation_0']['rmse']
    val_los = eval_res['validation_1']['rmse']


    plt.figure()
    plt.plot(list(range(n)), train_los, 'b', label='Train RMSE loss')
    plt.plot(list(range(n)), val_los, 'r', label='Val RMSE loss')

    plt.title(f'Train and val RMSE loss comp, n={n}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'n_estimators_choosing/with_lr_0003/losses_comp_{n}.png')



#lrs = [0.3, 0.1, 0.003, 0.01]
#for x in lrs:
#    lr_experiments(x)

n_estimators = [2000, 2500, 3000, 3500, 4000]
for x in n_estimators:
    n_estimators_experiments(x)










'''
#Training model
model = XGBRegressor(
    n_estimators_choosing=1500,
    learning_rate=0.1,
    max_depth=5,
    subsample=1.0,           
    colsample_bytree=1.0,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],
    verbose=True
)


#Checking val results
eval_res = model.evals_result()

train_los = eval_res['validation_0']['rmse']
val_los = eval_res['validation_1']['rmse']


#Graph losses
plt.plot(list(range(1500)), train_los, 'b', label='Train loss')
plt.plot(list(range(1500)), val_los, 'r', label='Val loss')

plt.title('Train and val loss comp')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('losses_comp_base.png')
'''







