for i in {1..3}; do
    bash bash_scripts/hRPL.sh --seed $i --experiment_name animals_greedy_cts_noiseOnTop0.1_pred_$i --loss pred --prediction_target enc --pred_lr_mult 10;
    bash bash_scripts/hRPL.sh --seed $i --experiment_name animals_greedy_cts_noiseOnTop0.1_freeze_pred_$i --loss pred --prediction_target enc --pred_lr_mult 10 --frozen_areas 0 1 2 3 4;
done