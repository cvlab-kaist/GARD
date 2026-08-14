source .venv/bin/activate
time CUDA_VISIBLE_DEVICES=3 python -m depth_anything_3.bench.evaluator --config run_configs/val/val_GARD_real_bench.yaml

