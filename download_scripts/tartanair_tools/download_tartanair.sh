# python download_training.py --rgb --cloudflare --output-dir ../data --unzip --only-left --only-easy

mkdir -p ./data/train/tartanair
python download_scripts/tartanair_tools/download_training.py --rgb    --cloudflare --output-dir ./data/train/tartanair --unzip --only-left --only-easy --huggingface
python download_scripts/tartanair_tools/download_training.py --depth  --cloudflare --output-dir ./data/train/tartanair --unzip --only-left --only-easy --huggingface

# python download_scripts/tartanair_tools/download_training.py --rgb    --cloudflare --output-dir ./data/train/tartanair --unzip --only-left --only-hard --huggingface
# python download_scripts/tartanair_tools/download_training.py --depth  --cloudflare --output-dir ./data/train/tartanair --unzip --only-left --only-hard --huggingface
# python download_training.py --seg --cloudflare --output-dir ../data --unzip
# python download_training.py --flow --cloudflare --output-dir ../data --unzip