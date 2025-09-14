.PHONY: test lint

test:
	python -m pytest tests/ -v
	python -c 'import doctest, re; f = lambda: None; f.__doc__ = re.search(r"```python\n(.*?)\n```", open("README.md").read(), re.DOTALL).group(1); doctest.run_docstring_examples(f, globals(), verbose=True)'

lint:
	python -m ruff check bengali_romanizer/ tests/
	python -m ruff format --check bengali_romanizer/ tests/