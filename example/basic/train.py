'''MLflow basic example
'''

import mlflow


if __name__ == '__main__':
    '''start mlflow session
    mlflow.start_run(run_id=None, experiment_id=None, run_name=None, nested=False)

    You don't have to worry about these arguments normally. If you want to define
    run_id, experiment_id and run_name explicitly and start your experiments nested
    stracture. See reference.
    https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run

    You can also use syntax like this:
    >>> with mlflow.start_run() as run:
    >>>     ...
    https://www.mlflow.org/docs/latest/python_api/mlflow.html#module-mlflow
    '''
    mlflow.start_run(run_name='basic_example')

    '''log parameter
    mlflow.log_param("set parameter name", "set parameter value")

    You can log parameter using log_param() function.
    '''
    mlflow.log_param('param_01', 'Hello, MLflow!!')
    mlflow.log_param('param_02', 123)

    ''' log metric basic
    mlflow.log_metric("set metric name", "set metric value")

    You can log metric using log_metric() function.
    '''
    mlflow.log_metric('metric_01', 0.234)
    mlflow.log_metric('metric_02', 0.567)

    '''log metric progress like TensorBoard
    mlflow.log_metric("set metric name", "set metric value", "set step")

    You can log metrics like TensorBoard using log_metric function with "step".
    '''
    for step, value in enumerate([0.6, 0.4, 0.2, 0.15, 0.13, 0.17]):
        mlflow.log_metric('loss', value, step)

    '''end mlflow session
    mlflow.end_run()

    If you use "with" syntax, this function do not need to type.
    https://www.mlflow.org/docs/latest/python_api/mlflow.html#module-mlflow
    https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.end_run
    '''
    mlflow.end_run()

    '''see logs
    You can see logs on browser with interactive UI.

    Type command on shell and access "http://localhost:5000" by default.
    > $ mlflow ui
    or
    > $ mlflow server

    https://mlflow.org/docs/latest/cli.html#mlflow-ui
    https://mlflow.org/docs/latest/cli.html#mlflow-server
    '''
    print('Enjoy MLflow :)')
