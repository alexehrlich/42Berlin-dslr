all: setup

setup:
	@virtualenv venv && pip install -r requirements.txt

clean:
	@rm -rf dataset/splitted
	@rm weights.pkl