bash stop_all.sh
mkdir -p pid
rm -f pid/train_v2.pid
#echo '======================================'>>pid/train_v2.pid
rm -rf log/*
python train_v2.py & echo $! >> pid/train_v2.pid
