bash train_robust_v2_stop.sh
mkdir -p pid
mkdir -p log
rm -f pid/train_robust_v2.pid

rm -rf log/rain_robust_v2_*
echo 'gpu_index : '${1:-0}
python ../train_robust_v2.py --log_to_file --gpu_index ${1:-0} & echo $! >> pid/train_robust_v2.pid
