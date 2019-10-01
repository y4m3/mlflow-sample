# mlflow-sample

## Environments

All codes runs in docker container. Makefile has useful commands to use docker.

## Build docker image

You can build docker image for this repository.

```bash
$ make init-docker
```

## How to use `make` commands

### Run docker container

You can run docker container.

```bash
$ make create-container
```

### Start stopped docker container

You can start stopped docekr container.

```bash
$ make start-container
```

### Remove docker container

You can stop and remove docker container.

```bash
$ make clean-container
```

## Special Thanks

- [MLflow Developers](https://github.com/mlflow/mlflow/graphs/contributors)
- [docker-science/cookiecutter-docker-science Developers](https://github.com/docker-science/cookiecutter-docker-science/graphs/contributors)

## Licence

MIT License
