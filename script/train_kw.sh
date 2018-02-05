export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES="5"

export PATH="$PATH:/usr/local/cuda-8.0/bin"
#export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
#export CUDA_VISIBLE_DEVICES="1"


python -u python/RL/train_kw.py
