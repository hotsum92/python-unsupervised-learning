build:
	docker build -t python-unsupervised-learning .

run:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning bash

logistic:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning python logistic.py

random-forest:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning python random-forest.py

light-gbm:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning python light-gbm.py
