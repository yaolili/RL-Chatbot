
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/local/cuda-8.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES='' 

export PATH="$PATH:/usr/local/cuda-8.0/bin"

python2.7 python/RL/test.py $1 $2 $3