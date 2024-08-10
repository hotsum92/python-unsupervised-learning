build:
	docker build -t python-unsupervised-learning .

run:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning bash

logistic:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning python logistic.py
