#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

lr_lst=(0.0007)
hiddenunits_lst=(32 64)
dropout_lst=(0.00001)
weight_decay_lst=(1e-5)
dropout_lst=(0.4 0.5)
skip_factor_lst=(1)


for hiddenunits in "${hiddenunits_lst[@]}"; do
    for skip_factor in "${skip_factor_lst[@]}"; do
        for dropout in "${dropout_lst[@]}"; do
            python main.py --dataset fb100 --sub_dataset Penn94 --simrank_file_name fb100-c_06-epison_0.01.pt --hiddenunits $hiddenunits --lr 0.0007 --dropout $dropout --weight_decay 0.0001 --delta 0.78 --epochs 200 --runs 5 --method simgnn --propa_mode post --skip_factor $skip_factor --display_step 1
        done
    done
done

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 