for i in {1..3}; do
    bash bash_scripts/mouse.sh --seed $i --experiment_name mouse_pred_$i --loss pred --prediction_target enc --pred_lr_mult 10;
    bash bash_scripts/mouse.sh --seed $i --experiment_name mouse_inv_sg_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 20.0 --decorr_coef 200.0;
done

bash bash_scripts/mouse.sh --seed $i --experiment_name mouse_random_$i --loss supervised --freeze;
bash bash_scripts/mouse.sh --seed $i --experiment_name mouse_supervised_$i --loss supervised;