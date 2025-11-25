for i in {1..3}; do
    bash bash_scripts/libri.sh --seed $i --experiment_name libri_delta8_pred_$i --loss pred --prediction_target enc --pred_steps 8;
    bash bash_scripts/libri.sh --seed $i --experiment_name libri_delta8_inv_sg_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 2.0 --decorr_coef 20.0 --pred_steps 8;
done

bash bash_scripts/libri.sh --seed $i --experiment_name libri_delta8_supervised_$i --loss supervised --lr 1e-3;
bash bash_scripts/libri.sh --seed $i --experiment_name libri_delta8_random_$i --loss supervised --freeze;