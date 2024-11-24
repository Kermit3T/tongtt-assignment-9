# Define your virtual environment and flask app
VENV = venv
PYTHON = python
PIP = $(VENV)\Scripts\pip
FLASK = $(VENV)\Scripts\flask
FLASK_APP = app.py

.PHONY: install run clean reinstall

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the Flask application
run:
	set FLASK_APP=$(FLASK_APP) && set FLASK_ENV=development && $(FLASK) run --port 3000

# Clean up virtual environment
clean:
	rmdir /s /q $(VENV)

# Reinstall all dependencies
reinstall: clean install