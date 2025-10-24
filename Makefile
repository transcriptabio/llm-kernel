.PHONY: test clean install dev

# Run all tests
test:
	python run_tests.py

# Install the package in development mode
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e .[dev]

# Clean up Python cache files
clean:
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Run linting
lint:
	ruff check src/

# Run formatting
format:
	ruff format src/

# Run linting and formatting
fix:
	ruff check --fix src/
	ruff format src/
