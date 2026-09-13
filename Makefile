.PHONY: install install-dev test test-all features verify-features lint format fetch-data train nowcast dashboard clean

# Install production dependencies
install:
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Run the fast suite (excludes repeated model refits)
test:
	pytest tests/ -m "not slow" -v --cov=src --cov-report=term-missing

# Run everything, including the slow model-refitting tests
test-all:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Generate a point-in-time feature panel for downstream models
features:
	python scripts/build_features.py

# Verify the feature panel is point-in-time (no future data in history)
verify-features:
	python scripts/build_features.py --verify

# Lint code
lint:
	ruff check src/ tests/ scripts/
	mypy src/

# Format code
format:
	black src/ tests/ scripts/ notebooks/
	ruff check --fix src/ tests/ scripts/

# Fetch latest FRED data
fetch-data:
	python scripts/fetch_data.py

# Train models on historical data
train:
	python scripts/train_model.py

# Run a single nowcast and print results
nowcast:
	python scripts/run_nowcast.py

# Launch Streamlit dashboard
dashboard:
	streamlit run src/dashboard/app.py

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name "htmlcov" -exec rm -rf {} +
	find . -name ".coverage" -delete
