.PHONY: install run dummy test simulate clean help

help:
	@echo "Targets:"
	@echo "  make install   Install package in editable mode with dev extras"
	@echo "  make run       Run the firewall server on :8000"
	@echo "  make dummy     Run the local dummy upstream LLM on :9000"
	@echo "  make test      Run the pytest suite"
	@echo "  make simulate  Run the standalone validation simulation"
	@echo "  make clean     Remove build, cache, and test artifacts"

install:
	pip install -e ".[dev]"

run:
	uvicorn llm_firewall.api.app:app --reload --port 8000

dummy:
	uvicorn llm_firewall.api.dummy_llm:app --reload --port 9000

test:
	pytest

simulate:
	python scripts/simulate.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
