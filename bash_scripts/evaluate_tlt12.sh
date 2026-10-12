save_dir="./tlt12_reps";

# save representations
for counter in {1..5}; do
    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_supervised_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name sup_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc;

    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_pred_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name RPL_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc;

    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_rplvic_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name RPLVIC_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc;

    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_inv_sg_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name IL_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target pred;

    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_ilvic_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name ILVIC_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target pred;

    python repl/scripts/save_tlt12_repr.py \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name rand_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc;
done;


for counter in {2..5}; do
    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_invZ_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name ILz_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc;

    python repl/scripts/save_tlt12_repr.py \
        --model_name tlt12_invZ_${counter} \
        --save_dir $save_dir \
        --batch_size 128 \
        --repr_file_name ILzEnc_${counter} \
        --encoder resnet \
        --enc_output_dim 256 \
        --use_bn \
        --enc_n_layers 4 \
        --resnet_n_channels 32 64 128 256 \
        --resnet_n_blocks 2 2 2 2 \
        --resnet_pool 0 1 1 1 \
        --resnet_stem_kernel_size 3 \
        --integrator lstm \
        --ctx_dim 256 \
        --predictor mlp \
        --pred_n_hidden_layers 1 \
        --pred_hidden_dim 1024 \
        --pred_steps 2 \
        --prediction_target enc \
        --repr_idx 0;
done;


python repl/scripts/eval_tlt12.py \
    --repr_path $save_dir;
python repl/scripts/eval_tlt12.py \
    --repr_path $save_dir --per_class;
python repl/scripts/eval_tlt12.py \
    --repr_path $save_dir --rot_probe_threshold 0.5;
