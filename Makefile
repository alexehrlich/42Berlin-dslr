all: setup

setup:
	@virtualenv venv && pip install -r requirements.txt