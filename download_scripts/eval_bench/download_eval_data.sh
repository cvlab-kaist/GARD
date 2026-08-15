# download the clean (undegraded) DA3-BENCH images from the official depth-anything HuggingFace dataset
# https://huggingface.co/datasets/depth-anything/DA3-BENCH
mkdir -p data/eval/da3_benchmark/clean
hf download depth-anything/DA3-BENCH --repo-type dataset --local-dir data/eval/da3_benchmark/clean

# extract each dataset zip in place, then remove the zip
for f in data/eval/da3_benchmark/clean/*.zip; do
    unzip -q "$f" -d data/eval/da3_benchmark/clean
    rm "$f"
done


# download our synthesized degraded counterpart of DA3-BENCH from HuggingFace
# https://huggingface.co/datasets/jinlovespho/GARD-eval-bench
hf download jinlovespho/GARD-eval-bench --repo-type dataset --local-dir data/eval/da3_benchmark/degraded --exclude "real_benchmark/*"


# download the real-world camera motion blur scenes (DeblurNeRF real captures, originally
# from the official DeblurNeRF google drive) from our HuggingFace mirror — gdown's
# per-file Google Drive downloads are unreliable at this file count and silently drop files
# https://huggingface.co/datasets/jinlovespho/GARD-eval-bench
hf download jinlovespho/GARD-eval-bench --repo-type dataset --local-dir data/eval --include "real_benchmark/*"


