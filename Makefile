IMAGE_NAME="egbert/data-science"

build: requirements.txt
	docker build -t $(IMAGE_NAME) .

start_jupyter: build
	docker run -it --rm -v $(PWD)/notebooks:/data_science/notebooks -p 8888:8888 --name data_science $(IMAGE_NAME)

requirements.txt: Pipfile Pipfile.lock
	pipenv lock -r > requirements.txt

.PHONY: build start_jupyter requirements.txt