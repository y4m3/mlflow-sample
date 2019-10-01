# Mlflow Example Basic Usage

## You have to learn in this sample code

- How to start `mlflow` session
- How to log parameters
- How to log metrics
  - Simple metric
  - Metric progress (like TensorBoard)

## The most simple sample

```python
import mlflow

# start mlflow session
mlflow.start_run()

# log parameter
mlflow.log_param('sample_param', 1)

# log metric
mlflow.log_metric('sample_metric', 0.5)

# log metric progress like TensorBoard
for step, metric_value in enumerate([0.6, 0.4, 0.2]):
    mlflow.log_metric('sample_metric_progress', metric_value, step)

# end mlflow session
mlflow.end_run()
```
