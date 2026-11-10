"""Overnight neural search on the GPU.

    .venv\\Scripts\\python.exe run_gpu.py

Runs sklearn's HalvingRandomSearchCV over preprocessing + transformer settings
(config.NEURAL_SEARCH / NEURAL_SEARCH_SPACE), refits the winner on the full
training set and writes data/submission_neural.csv. Extra arguments are passed
through, e.g. `--sample-size 3000` for a quick test.
"""
import os
import sys

# Quiet, offline-safe Hugging Face setup (models are already in the local cache).
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

for stream in (sys.stdout, sys.stderr):
    # Progress bars use block characters; never let the console encoding crash the run.
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

from classification_model import main


# NEURAL NETWORK DISABLED (needs an NVIDIA GPU): uncomment to use it.
# if __name__ == '__main__':
#     import torch

#     if not torch.cuda.is_available():
#         sys.exit('CUDA is not available. Run this with .venv\\Scripts\\python.exe '
#                  '(PyTorch CUDA build), not the system Python.')
#     print(f'CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__} '
#           f'(CUDA {torch.version.cuda})', flush=True)
#     main(['--model', 'neural_network', '--halving-search', *sys.argv[1:]])
