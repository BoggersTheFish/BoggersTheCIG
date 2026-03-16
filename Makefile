# Full-TS Cognitive Architecture - Makefile
# Windows: use `make` via WSL or run commands manually

.PHONY: install db redis test run run-continuous api clean

install:
	pip install -r requirements.txt
	pip install -e .

db:
	docker-compose up -d memgraph

redis:
	docker-compose up -d redis

up:
	docker-compose up -d

init:
	python -c "from src.concept_graph import ConceptGraph; ConceptGraph().init_schema()"

test:
	pytest tests/ -v

run:
	python src/main.py --input "Explain quantum physics"

run-continuous:
	celery -A src.tasks worker --loglevel=info &
	python src/main.py --mode=continuous

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0

check:
	python src/deploy.py --check

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
