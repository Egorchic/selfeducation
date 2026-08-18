## Базовая конфигурация

```text
n_estimators = 1500
max_depth = 5
subsample = 1.0
colsample_bytree = 1.0
```

## Сравнение разных learning_rate

![losses_png](learning_rate_choosing/losses_comp_0.3.png)

![losses_png](learning_rate_choosing/losses_comp_0.1.png)

![losses_png](learning_rate_choosing/losses_comp_0.01.png)

![losses_png](learning_rate_choosing/losses_comp_0.003.png)

При learning_rate=0.3 модель быстро обучается, но примерно после нескольких сотен деревьев validation RMSE перестаёт уменьшаться и начинает расти, в то время как train RMSE продолжает снижаться. Это говорит о переобучении

При learning_rate=0.1 переобучение выражено слабее: validation RMSE постепенно выходит на плато, тогда как train RMSE продолжает снижаться.

При learning_rate=0.01 и 0.003 обе метрики к 1500-й итерации всё ещё уменьшаются, поэтому для этих значений необходимо увеличить число деревьев, прежде чем сравнивать их минимальный validation RMSE.