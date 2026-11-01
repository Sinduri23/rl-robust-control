# Results

30 seeded episodes per setting; same seeds for every controller.

## Nominal conditions

| controller              |   survival |   success_rate |   angle_err |   switch_rate |
|:------------------------|-----------:|---------------:|------------:|--------------:|
| PID                     |        500 |            100 |       0.196 |         0.633 |
| PPO (no rand.)          |        500 |            100 |       0.284 |         0.695 |
| PPO (rand. mass/length) |        500 |            100 |       0.272 |         0.702 |
| PPO (full rand.)        |        500 |            100 |       0.47  |         0.692 |

## Combined hard scenario (3x mass, 1.5x length, noise 0.1, pushes 1.0)

| controller              |   survival |   success_rate |   angle_err |   switch_rate |
|:------------------------|-----------:|---------------:|------------:|--------------:|
| PID                     |     47.267 |              0 |       3.934 |         0.326 |
| PPO (no rand.)          |     36.633 |              0 |       4.3   |         0.311 |
| PPO (rand. mass/length) |     58.233 |              0 |       4.137 |         0.353 |
| PPO (full rand.)        |     80.867 |              0 |       4.212 |         0.403 |

## Average over all single-disturbance settings

| controller              |   survival |   success_rate |
|:------------------------|-----------:|---------------:|
| PID                     |      424.6 |           80.4 |
| PPO (full rand.)        |      419.5 |           77.8 |
| PPO (no rand.)          |      362.5 |           66.2 |
| PPO (rand. mass/length) |      422.7 |           79.8 |
