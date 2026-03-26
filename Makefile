ingest:
	python backend/run_ingestion.py --reset

ingest-pack:
	python backend/run_ingestion.py --reset --use-pack

ingest-report:
	python backend/run_ingestion.py --use-pack --save-report backend/resources/ingestion_report.json

resources-validate:
	python backend/run_ingestion.py --use-pack --validate-resources --save-report backend/resources/ingestion_report.json

run:
	docker compose up --build

eval:
	python backend/eval/run_eval.py

test:
	pytest -q
