.PHONY: install generate train serve dashboard test clean

# Install all dependencies
install:
	pip install -r requirements.txt

# Generate synthetic CDR + CRM data (50K subscribers)
generate:
	python src/data/generate_synthetic.py

# Run ETL pipeline: raw data → feature-engineered parquet
etl:
	python src/data/etl_pipeline.py

# Train XGBoost model + generate SHAP + evaluation charts
train: generate etl
	python src/models/train.py
	python src/models/evaluate.py

# Start Flask REST API (port 5000)
serve:
	python src/api/app.py

# Start Streamlit dashboard (port 8501)
dashboard:
	streamlit run src/dashboard/streamlit_app.py

# Run API smoke tests
test:
	pytest tests/ -v

# Load data into SQLite DB
db:
	python src/data/db_loader.py

# Clean generated artifacts
clean:
	rm -rf data/raw/*.csv data/processed/*.parquet
	rm -rf src/models/artifacts/*.pkl src/models/artifacts/*.json
	rm -rf evaluation/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
