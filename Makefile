all: setup

setup:
	@virtualenv venv && source venv/bin/activate . && pip3 install -r requirements.txt

clean:
	@rm -rf dataset/splitted
	@rm weights.pkl