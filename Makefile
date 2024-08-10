credit-card.csv:
	curl -o credit-card.csv https://raw.githubusercontent.com/aapatel09/handson-unsupervised-learning/master/datasets/credit_card_data/correlationMatrix.csv

build:
	docker build -t python-unsupervised-learning .

run:
	docker run -it --rm --gpus all -u $(id -u):$(id -g) -v ./:/usr/src python-unsupervised-learning bash
