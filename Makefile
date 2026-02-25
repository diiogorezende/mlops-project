include .envs/.mlflow-common
include .envs/.mlflow-dev
include .envs/.postgres
export

DOCKER_COMPOSE_RUN = docker-compose run --rm mlflow-server

build:
	docker-compose build
up:
	docker-compose up -d
down:
	docker-compose down
exec-in: up
	docker exec -it local-mlflow-tracking-server bash