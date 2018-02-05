export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
export CUDA_VISIBLE_DEVICES="1"

if [ $1 == "S2S" ]; then
	./script/test.sh model/Seq2Seq/model-77 $2 $3
elif [ $1 == 'RL' ]; then
	./script/test_RL.sh model/RL/model-56-3000 $2 $3
else
	./script/test_RL.sh model/RL/model-56-3000 $2 $3
fi
