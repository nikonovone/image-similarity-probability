.PHONY: *

PYTHON_EXEC := python

CLEARML_PROJECT_NAME := "Image Similarity Probability"
CLEARML_DATASET_NAME := original


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python

run_training:
	poetry run $(PYTHON_EXEC) -m src.train
