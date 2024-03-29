.PHONY: clean clean-pyc docs help init init-docker create-container start-container jupyter jupyterlab mlflow test lint profile clean clean-docker clean-container clean-image sync-from-source sync-to-source
.DEFAULT_GOAL := help

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import os, re, sys

if os.environ['TARGET']:
    target = os.environ['TARGET']
    is_in_target = False
    for line in sys.stdin:
        match = re.match(r'^(?P<target>{}):(?P<dependencies>.*)?## (?P<description>.*)$$'.format(target).format(target), line)
        if match:
            print("target: %-20s" % (match.group("target")))
            if "dependencies" in match.groupdict().keys():
                print("dependencies: %-20s" % (match.group("dependencies")))
            if "description" in match.groupdict().keys():
                print("description: %-20s" % (match.group("description")))
            is_in_target = True
        elif is_in_target == True:
            match = re.match(r'^\t(.+)', line)
            if match:
                command = match.groups()
                print("command: %s" % (command))
            else:
                is_in_target = False
else:
    for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
            target, help = match.groups()
            print("%-20s %s" % (target, help))
endef

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef

###########################################################################################################
## VARIABLES
###########################################################################################################
export DOCKER=docker
export TARGET=
export PWD=$(shell pwd)
export PRINT_HELP_PYSCRIPT
export START_DOCKER_CONTAINER
export PROJECT_NAME=ml
export IMAGE_NAME=$(PROJECT_NAME)-image
export CONTAINER_NAME=$(PROJECT_NAME)-container
export DATA_SOURCE=
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export MLFLOW_HOST_PORT=5000
export MLFLOW_CONTAINER_PORT=5000
export PYTHON=python
export DOCKERFILE=docker/Dockerfile

###########################################################################################################
## ADD TARGETS SPECIFIC TO "mlflow-sample"
###########################################################################################################


###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help: ## show this message
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init: init-docker sync-from-source ## initialize repository for traning

sync-from-source: ## download data data source to local envrionment

init-docker: ## initialize docker image
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

init-docker-no-cache: ## initialize docker image without cachhe
	$(DOCKER) build --no-cache -t $(IMAGE_NAME) -f $(DOCKERFILE) .

sync-to-source: ## sync local data to data source
	echo "no sync target for url data source..."

create-container: ## create docker container
	$(DOCKER) run -itd -v $(PWD):/work -p 2222:22 -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) -p $(MLFLOW_HOST_PORT):${MLFLOW_CONTAINER_PORT} --name $(CONTAINER_NAME) $(IMAGE_NAME) /bin/bash

start-container: ## start docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."

jupyter: ## start Jupyter Notebook server
	jupyter-notebook --allow-root --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}

jupyterlab: ## start Jupyter Lab server
	jupyter lab --allow-root --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}

mlflow: ## start MLflow server
	mlflow server --port ${MLFLOW_CONTAINER_PORT} --host 0.0.0.0

test: ## run test cases in tests directory
	$(PYTHON) -m unittest discover

lint: ## check style with flake8
	flake8 modules

profile: ## show profile of the project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
	@echo "JUPYTER_PORT: `$(DOCKER) port $(CONTAINER_NAME)`"
	@echo "DATA_SOURE: $(DATA_SOURCE)"

clean: clean-pyc clean-docker ## remove all artifacts

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) stop $(CONTAINER_NAME)
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)

# This Makefile cites cookiecutter-docker-science and modified.
# https://github.com/docker-science/cookiecutter-docker-science/blob/master/%7B%7B%20cookiecutter.project_slug%20%7D%7D/Makefile
