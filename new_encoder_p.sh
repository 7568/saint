cd new_encoder/sh
bash train_robust_v2_stop.sh
cd ../../
cd new_encoder_2/sh
bash train_robust_v2_stop.sh
cd ../../
cd new_encoder_3/sh
bash train_robust_v2_stop.sh
cd ../../
rm -rf new_encoder_2
rm -rf new_encoder_3
cp -r new_encoder new_encoder_2
cp -r new_encoder new_encoder_3

cd new_encoder/sh
bash train_robust_v2_restart.sh 5 2
cd ../../
cd new_encoder_2/sh
bash train_robust_v2_restart.sh 6 2
cd ../../
cd new_encoder_3/sh
bash train_robust_v2_restart.sh 7 3
