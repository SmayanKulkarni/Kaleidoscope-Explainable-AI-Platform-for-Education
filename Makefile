# ─────────────────────────────────────────────────────────────────────────────
# XAI Learning Recommendation System — Developer Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:  make <target>
# Requires: Python 3.11+, Docker, Terraform, AWS CLI
# ─────────────────────────────────────────────────────────────────────────────

PYTHON      ?= python
PIP         ?= pip
VENV_PYTHON  = .venv/Scripts/python
VENV_PIP     = .venv/Scripts/pip

.PHONY: help install install-dev venv \
        run run-docker \
        test test-store test-ml test-smoke test-all lint \
        db-migrate \
        docker-build docker-up docker-down \
        tf-init tf-plan tf-apply tf-destroy \
        models-upload models-download \
        retrain reload \
        clean

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "XAI Learning Recommendation System"
	@echo "==================================="
	@echo ""
	@echo "  Dev setup:"
	@echo "    make venv            Create .venv"
	@echo "    make install         Install all deps into .venv"
	@echo "    make install-dev     Install + dev tools (ruff, pytest)"
	@echo ""
	@echo "  Run:"
	@echo "    make run             Start API locally (SQLite fallback)"
	@echo "    make run-docker      Start full stack via docker-compose"
	@echo ""
	@echo "  Test:"
	@echo "    make test            Run all tests except API smoke"
	@echo "    make test-all        Run full test suite"
	@echo "    make lint            Ruff lint check"
	@echo ""
	@echo "  DB:"
	@echo "    make db-migrate      Run Alembic migrations (requires DATABASE_URL)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build    Build Docker image"
	@echo "    make docker-up       Start docker-compose stack"
	@echo "    make docker-down     Stop and remove containers"
	@echo ""
	@echo "  Terraform:"
	@echo "    make tf-init         terraform init"
	@echo "    make tf-plan         terraform plan"
	@echo "    make tf-apply        terraform apply"
	@echo "    make tf-destroy      terraform destroy (destructive!)"
	@echo ""
	@echo "  MLOps:"
	@echo "    make models-upload   Upload models/ to S3"
	@echo "    make models-download Download models/ from S3"
	@echo "    make retrain         Trigger retrain on live server"
	@echo "    make reload          Hot-reload model on live server"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

venv:
	$(PYTHON) -m venv .venv
	@echo "Activate with: .venv\\Scripts\\activate"

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install-dev: install
	$(VENV_PIP) install ruff pytest pytest-asyncio httpx

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	$(VENV_PYTHON) -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

run-docker: docker-up

# ── Lint & Test ───────────────────────────────────────────────────────────────

lint:
	$(VENV_PYTHON) -m ruff check backend/ --select E,F,W,I --ignore E501

test:
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short --ignore=tests/test_api_smoke.py

test-store:
	$(VENV_PYTHON) -m pytest tests/test_event_store.py tests/test_feedback_store.py tests/test_db_config.py -v

test-ml:
	$(VENV_PYTHON) -m pytest tests/test_implicit_aggregator.py tests/test_engagement_model.py -v

test-smoke:
	$(VENV_PYTHON) -m pytest tests/test_api_smoke.py -v --tb=short

test-all:
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short

# ── Database Migrations ───────────────────────────────────────────────────────

db-migrate:
	@echo "Running Alembic migrations against DATABASE_URL=$(DATABASE_URL)"
	$(VENV_PYTHON) -m alembic upgrade head

db-revision:
	@read -p "Migration message: " msg; \
	$(VENV_PYTHON) -m alembic revision --autogenerate -m "$$msg"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t xai-rec-api:local .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

# ── Terraform ─────────────────────────────────────────────────────────────────

tf-init:
	terraform -chdir=infra init

tf-plan:
	terraform -chdir=infra plan

tf-apply:
	terraform -chdir=infra apply

tf-destroy:
	@echo "WARNING: This will DESTROY all cloud infrastructure!"
	terraform -chdir=infra destroy

# ── MLOps ─────────────────────────────────────────────────────────────────────

models-upload:
	@test -n "$(AWS_S3_BUCKET)" || (echo "Set AWS_S3_BUCKET first"; exit 1)
	aws s3 cp models/ s3://$(AWS_S3_BUCKET)/models/ --recursive

models-download:
	@test -n "$(AWS_S3_BUCKET)" || (echo "Set AWS_S3_BUCKET first"; exit 1)
	aws s3 cp s3://$(AWS_S3_BUCKET)/models/ models/ --recursive

retrain:
	@test -n "$(API_HOST)" || (echo "Set API_HOST=http://your-ec2-ip:8000"; exit 1)
	curl -sf -X POST "$(API_HOST)/mlops/retrain" | python -m json.tool

reload:
	@test -n "$(API_HOST)" || (echo "Set API_HOST=http://your-ec2-ip:8000"; exit 1)
	curl -sf -X POST "$(API_HOST)/mlops/reload" | python -m json.tool

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info
