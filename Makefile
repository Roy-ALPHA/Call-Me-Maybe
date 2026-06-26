install:
	uv sync

run:
	uv run -m src $(ARGS)

debug:
	python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + && find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run flake8 src && uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
