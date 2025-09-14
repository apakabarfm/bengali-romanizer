.PHONY: test

test:
	python -m pytest tests/ -v
	python -c 'import doctest, re; f = lambda: None; f.__doc__ = re.search(r"```python\n(.*?)\n```", open("README.md").read(), re.DOTALL).group(1); doctest.run_docstring_examples(f, globals(), verbose=True)'