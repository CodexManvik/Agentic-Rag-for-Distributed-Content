ingest:
	python backend/run_ingestion.py --reset

run:
	docker compose up --build

eval:
	python backend/eval/run_eval.py

test:
	pytest -q
