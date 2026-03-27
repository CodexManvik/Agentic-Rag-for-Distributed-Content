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

eval-dev:
	python backend/eval/run_eval.py --dataset backend/eval/dataset_dev.jsonl

eval-hidden:
	python backend/eval/run_eval.py --dataset backend/eval/dataset_hidden.jsonl

eval-split:
	python backend/eval/prepare_dataset_splits.py --input backend/eval/dataset.jsonl --dev-output backend/eval/dataset_dev.jsonl --hidden-output backend/eval/dataset_hidden.jsonl

eval-candidates:
	python backend/eval/generate_candidate_dataset.py --output backend/eval/candidate_dataset.jsonl

eval-build-demo:
	python backend/eval/build_demo_matrix_dataset.py --base backend/eval/dataset.jsonl --candidate backend/eval/candidate_dataset.jsonl --output backend/eval/dataset_dev.jsonl

eval-matrix-check:
	python backend/eval/check_matrix_coverage.py --dataset backend/eval/dataset_dev.jsonl --target backend/eval/eval_matrix_target.json

demo-prewarm:
	python backend/scripts/prewarm_demo.py --queries 3

demo-cache:
	python backend/scripts/build_demo_cache.py --backend-url http://localhost:8000/chat --output backend/resources/demo_cached_answers.json

test:
	pytest -q
