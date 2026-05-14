.PHONY: install train api frontend test lint docker

install:
	pip install -r requirements-dev.txt

train:
	python -m ml.train

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

test:
	pytest

lint:
	ruff check .

docker:
	docker compose up --build
