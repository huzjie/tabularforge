.PHONY: doctor test bench serve clean

doctor:
	python -m tabularforge doctor

test:
	python -m unittest discover -s tabularforge/tests -t .

bench:
	python -m tabularforge bench

serve:
	python -m tabularforge serve --host 0.0.0.0 --port 8000

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
