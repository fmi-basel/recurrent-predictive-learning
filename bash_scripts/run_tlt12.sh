for i in {1..5}; do
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_pred_$i --loss pred --prediction_target enc --pred_steps 2 --pred_lr_mult 10 --use_scheduler --scheduler_warmup_epochs 2 --epochs 50;
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_inv_sg_$i --loss inv --prediction_target pred --pred_steps 2 --pull_coef 13.08 --push_coef 39.16 --decorr_coef 83.46 --use_scheduler --scheduler_warmup_epochs 2 --epochs 50;
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_rplvic_$i --loss rplvic --prediction_target enc --pred_steps 2 --pull_coef 0.03 --push_coef 0.9 --decorr_coef 18.0 --use_scheduler --scheduler_warmup_epochs 2 --epochs 20;
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_ilvic_$i --loss ilvic --prediction_target pred --pred_steps 2 --pull_coef 13.08 --push_coef 39.16 --decorr_coef 83.46 --pull_coef_extra_reg 0.03 --push_coef_extra_reg 0.9 --decorr_coef_extra_reg 18.0 --use_scheduler --scheduler_warmup_epochs 2 --epochs 20;
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_supervised_$i --loss supervised --lr 1e-3 --epochs 50;
done
for i in {2..5}; do
    bash bash_scripts/tlt12.sh --seed $i --experiment_name tlt12_invZ_$i --loss inv --prediction_target enc --no_sg  --pred_steps 2 --pull_coef 0.03 --push_coef 0.9 --decorr_coef 18.0 --use_scheduler --scheduler_warmup_epochs 2 --epochs 20;
done
