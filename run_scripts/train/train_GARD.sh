source .venv/bin/activate

NUM_GPUS=1
CUDA=2

export SERVER=${SERVER}
export CUDA=${CUDA}
# put the repo root on PYTHONPATH so RAE/src/train.py can resolve cross-package
# imports (e.g. `RAE.src...`, `gard...`) regardless of its own script directory
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"


CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train.py \
  --config run_configs/train/train_GARD.yaml