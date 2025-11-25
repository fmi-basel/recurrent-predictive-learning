for i in {1..3}; do
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_pred_$i --loss pred --prediction_target enc --pred_lr_mult 10;
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_inv_sg_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 20.0 --decorr_coef 200.0;
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_pred_ctx_target_$i --loss pred --prediction_target ctx --pred_lr_mult 10;
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_inv_sg_nr_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 0.0 --decorr_coef 0.0;
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_invC_$i --loss inv --prediction_target ctx --pull_coef 1.0 --push_coef 20.0 --decorr_coef 200.0;
    bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_invZ_$i --loss inv --prediction_target enc --pull_coef 1.0 --push_coef 20.0 --decorr_coef 200.0;
done

bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_supervised_$i --loss supervised --lr 1e-3;
bash bash_scripts/moving_animals.sh --seed $i --experiment_name animals_cts_noiseOnTop0.1_random_$i --loss supervised --freeze;