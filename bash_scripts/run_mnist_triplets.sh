for i in {1..3}; do
    bash bash_scripts/mnist_triplets.sh --seed $i --experiment_name mnist_triplets_pred_$i --loss pred --prediction_target enc;
    bash bash_scripts/mnist_triplets.sh --seed $i --experiment_name mnist_triplets_inv_sg_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 1.0 --decorr_coef 10.0;
done

bash bash_scripts/mnist_triplets.sh --seed $i --experiment_name mnist_triplets_supervised_$i --loss supervised;
bash bash_scripts/mnist_triplets.sh --seed $i --experiment_name mnist_triplets_random_$i --loss supervised --freeze;