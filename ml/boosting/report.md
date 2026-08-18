## Базовая конфигурация

```text
learning_rate = 0.1
n_estimators = 1500
max_depth = 5
```

## Сравнение разных learning_rate

![losses_png](learning_rate_choosing/losses_comp_0.3.png)

![losses_png](learning_rate_choosing/losses_comp_0.1.png)

![losses_png](learning_rate_choosing/losses_comp_0.01.png)

![losses_png](learning_rate_choosing/losses_comp_0.003.png)

По графикам видно, что при learning_rate=0.3, 0.1 модель не может дообучится, так как train_loss стагнирует, то есть попала в локальный минимум из которого не может выбраться.

По графикам видно, что при learning_rate=0.01, 0.003, видно, что модель не дообучается и нужно больше деревьев