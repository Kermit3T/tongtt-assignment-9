# Define your virtual environment and flask app
VENV = venv
PYTHON = python
VENV_PYTHON = $(VENV)\Scripts\python
VENV_PIP = $(VENV)\Scripts\pip
FLASK = $(VENV)\Scripts\flask
FLASK_APP = app.py

.PHONY: install run clean reinstall

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	.\$(VENV)\Scripts\activate && \
	$(VENV_PIP) install -r requirements.txt

# Run the Flask application
run:
	.\$(VENV)\Scripts\activate && \
	set FLASK_APP=$(FLASK_APP) && \
	set FLASK_ENV=development && \
	$(FLASK) run --port 3000

# Clean up virtual environment
clean:
	if exist $(VENV) rmdir /s /q $(VENV)

# Reinstall all dependencies
reinstall: clean install