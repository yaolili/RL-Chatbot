export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES="1"

export PATH="$PATH:/usr/local/cuda-8.0/bin"

python -u python/RL/train.py
