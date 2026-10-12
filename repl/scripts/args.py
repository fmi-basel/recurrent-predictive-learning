from repl.data._valid_names_lists import DATASET_NAMES, MNIST_SEQTYPES
from repl.networks._valid_names_lists import ENCODER_NAMES, INTEGRATOR_NAMES, PREDICTOR_NAMES
from repl.modules._valid_names_lists import LOSS_NAMES


def add_reproducibility_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic training")


def size_tuple(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        return None


def add_model_args(parser):
    add_encoder_args(parser)
    add_context_integrator_args(parser)
    add_predictor_args(parser)


def add_greedy_model_args(parser):
    parser.add_argument("--n_areas", type=int, default=None, help="number of areas")
    parser.add_argument("--frozen_areas", type=int, nargs='+', default=None, help="frozen areas (0-indexed)")
    add_greedy_encoder_args(parser)
    add_greedy_integrator_args(parser)
    add_greedy_predictor_args(parser)


def add_encoder_args(parser):
    parser.add_argument("--encoder", choices=ENCODER_NAMES, default="mlp", help="encoder architecture")
    parser.add_argument("--enc_output_dim", type=int, default=512, help="output size of the encoder network")
    parser.add_argument("--enc_n_layers", type=int, default=2, help="number of layers in the encoder")
    parser.add_argument("--enc_kernel_size", type=size_tuple, nargs='+', default=None, help="kernel size of the encoder")
    parser.add_argument("--enc_stride", type=size_tuple, nargs='+', default=None, help="stride of the encoder")
    parser.add_argument("--enc_padding", type=size_tuple, nargs='+', default=None, help="padding of the encoder")
    parser.add_argument("--enc_pool_size", type=size_tuple, nargs='+', default=None, help="pool kernel size of the encoder")
    parser.add_argument("--resnet_n_channels", type=int, nargs='+', default=None, help="number of channels in the Custom ResNet encoder")
    parser.add_argument("--resnet_n_blocks", type=int, nargs='+', default=None, help="number of blocks in the Custom ResNet encoder")
    parser.add_argument("--resnet_pool", type=int, nargs='+', default=None, help="pooling for the Custom ResNet encoder")
    parser.add_argument("--resnet_stem_kernel_size", type=int, default=3, help="kernel size of the stem convolution in the Custom ResNet encoder")
    parser.add_argument("--flatten_enc_output", action="store_true", help="flatten the output of the encoder")
    parser.add_argument("--use_bn", action="store_true", help="use batch normalization in the encoder")
    

def add_greedy_encoder_args(parser):
    parser.add_argument("--area_encoders_kind", choices=ENCODER_NAMES, default="mlp", help="encoder architecture")
    parser.add_argument("--area_enc_dims", type=int, nargs='+', default=[512], help="output size of the encoder network")
    parser.add_argument("--area_enc_n_layers", type=int, default=1, help="number of layers in the encoder")
    parser.add_argument("--area_enc_kernel_sizes", type=size_tuple, nargs='+', default=None, help="kernel size of the encoder")
    parser.add_argument("--area_enc_strides", type=size_tuple, nargs='+', default=None, help="stride of the encoder")
    parser.add_argument("--area_enc_paddings", type=size_tuple, nargs='+', default=None, help="padding of the encoder")
    parser.add_argument("--area_enc_pool_sizes", type=size_tuple, nargs='+', default=None, help="pool kernel size of the encoder")
    parser.add_argument("--flatten_area_enc_output", action="store_true", help="flatten the output of the encoder")
    parser.add_argument("--area_enc_bn", action="store_true", help="use batch normalization in the encoder")


def add_context_integrator_args(parser):
    parser.add_argument("--integrator", choices=INTEGRATOR_NAMES, default="identity", help="context integrator architecture")
    parser.add_argument("--ctx_dim", type=int, default=512, help="output size of the context integrator network")
    parser.add_argument("--ctx_n_layers", type=int, default=1, help="number of layers in the context integrator network")


def add_greedy_integrator_args(parser):
    parser.add_argument("--area_integrators_kind", choices=INTEGRATOR_NAMES, default="rnn", help="context integrator architecture")
    parser.add_argument("--area_ctx_dims", type=int, nargs='+', default=[512], help="output size of the context integrator network")
    parser.add_argument("--area_ctx_n_layers", type=int, default=1, help="number of layers in the context integrator network")


def add_prediction_args(parser):
    parser.add_argument("--pred_steps", type=int, default=12, help="number of prediction steps")
    parser.add_argument("--dense_prediction", action="store_true", help="use dense prediction")
    parser.add_argument("--prediction_target", choices=["enc", "ctx", "pred"], default="enc", help="prediction target (encoder, context or predictor output)")
    parser.add_argument("--pred_target_dim_override", type=int, default=0, help="override the output size of the predictor network (only used for inv and ILwVIC loss)")


def add_predictor_args(parser):
    parser.add_argument("--predictor", choices=PREDICTOR_NAMES, default="mlp", help="predictor architecture")
    parser.add_argument("--pred_n_hidden_layers", type=int, default=1, help="number of hidden layers in the predictor network")
    parser.add_argument("--pred_hidden_dim", type=int, default=512, help="hidden size of the predictor network (only used for an MLP predictor)")
    add_prediction_args(parser)


def add_greedy_predictor_args(parser):
    parser.add_argument("--area_predictors_kind", choices=PREDICTOR_NAMES, default="mlp", help="predictor architecture")
    parser.add_argument("--area_pred_n_hidden_layers", type=int, default=1, help="number of hidden layers in the predictor network")
    parser.add_argument("--area_pred_hidden_dims", type=int, nargs='+', default=[512], help="hidden size of the predictor network (only used for an MLP predictor)")
    add_prediction_args(parser)

def add_data_args(parser):
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="stl10", help="dataset")
    parser.add_argument("--val_size", type=float, default=0.0, help="validation size")
    parser.add_argument("--data_input_dir", type=str, default="datasets", help="path to the data directory")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers for data loading")
    parser.add_argument("--seq_len", type=int, default=None, help="sequence length")
    parser.add_argument("--grayscale", action="store_true", help="use grayscale images")
    parser.add_argument("--flatten_images", action="store_true", help="flatten images (only applies to small image datasets)")
    parser.add_argument("--mnist_seqtype", choices=MNIST_SEQTYPES, default="triplets", help="sequence type for MNIST datasets")
    parser.add_argument("--spritevid_max_sprites", type=int, default=16, help="maximum number of sprites in the SpriteVideo dataset")
    parser.add_argument("--spritevid_exclude_latent_regions", action="store_true", help="exclude latent regions during training for the SpriteVideo dataset (for testing generalization)")
    parser.add_argument("--spritevid_discretize_latents", action="store_true", help="discretize latents for the SpriteVideo dataset")
    parser.add_argument("--spritevid_noise_type", choices=["gaussian", "salt_pepper"], default=None, help="noise type for the SpriteVideo dataset")
    parser.add_argument("--sprite_noise_on_top", action="store_true", help="If passed, the noise will be added to the whole frame including the sprites; otherwise, noise will be added only to the background")
    parser.add_argument("--spritevid_noise_level", type=float, default=0.1, help="noise level for the SpriteVideo dataset")
    parser.add_argument("--spritevid_frozen_noise", action="store_true", help="freeze noise across frames for the SpriteVideo dataset")
    parser.add_argument("--spritevid_grid_enabled", action="store_true", help="enable grid for the SpriteVideo dataset")
    parser.add_argument("--spritevid_frozen_grid", action="store_true", help="freeze grid across frames for the SpriteVideo dataset")
    parser.add_argument("--spritevid_occlude_n_frames", type=int, default=0, help="number of frames to occlude in the SpriteVideo dataset")
    parser.add_argument("--num_sequences", type=int, default=10000, help="number of sequences for MNIST dataset")
    parser.add_argument("--inter_trial_interval", type=int, default=0, help="inter-trial interval (only for MNIST dataset)")


def add_optimization_args(parser):
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--pred_lr_mult", type=float, default=1.0, help="learning rate multiplier for predictor")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="optimizer to use")
    parser.add_argument("--use_scheduler", action="store_true", help="use scheduler")
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=10, help="number of warmup epochs for the scheduler")


def add_training_args(parser):
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--freeze", action="store_true", help="freeze the model and only train the classifier")


def add_logging_args(parser):
    parser.add_argument("--nolog", action="store_true", help="stop log to wandb")
    parser.add_argument("--experiment_name", type=str, default="default", help="name of the experiment")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=0, help="save checkpoint every n epochs. 0 to disable")


def add_criterion_args(parser):
    parser.add_argument("--loss", choices=LOSS_NAMES, default="pred", help="loss function to use")
    parser.add_argument("--regularize_over", choices=["batch", "batch+time", "time"], default="batch+time", help="regularize over batch, time or batch+time")
    parser.add_argument("--discount_factor", type=float, default=0.0, help="discount factor")
    parser.add_argument("--pull_coef", type=float, default=1.0, help="pull coefficient")
    parser.add_argument("--push_coef", type=float, default=1.0, help="push coefficient")
    parser.add_argument("--decorr_coef", type=float, default=10.0, help="decorrelation coefficient")
    parser.add_argument("--pred_loss_type", choices=["cosine", "l2"], default="l2", help="predictor loss type (only used for Pred loss)")
    parser.add_argument("--no_sg", action="store_true", help="do not use stop-gradient for the target network (only applies to non-contrastive SSL losses)")
    parser.add_argument("--pull_coef_extra_reg", type=float, default=1.0, help="pull coefficient for extra regularization (only applies to ILwVIC loss)")
    parser.add_argument("--push_coef_extra_reg", type=float, default=1.0, help="push coefficient for extra regularization (only applies to ILwVIC loss)")
    parser.add_argument("--decorr_coef_extra_reg", type=float, default=1.0, help="decorrelation coefficient for extra regularization (only applies to ILwVIC loss)")


def add_validation_args(parser):
    parser.add_argument("--evaluate_concat_features", action="store_true", help="evaluate concatenated features")
    parser.add_argument("--val_batch_size", type=int, default=256, help="batch size")


def add_online_eval_args(parser):
    parser.add_argument("--online_task", choices=["seq2label", "seq2seq", "multitask"], default="seq2label", help="online classification task")
    parser.add_argument("--online_input", choices=["enc", "ctx", "pred"], default="enc", help="input to the online classifier")
    parser.add_argument("--online_lr", type=float, default=1e-3, help="learning rate for online evaluation")
    parser.add_argument("--online_weight_decay", type=float, default=1e-5, help="weight decay for online evaluation")
    parser.add_argument("--online_single_timestep_readout", action="store_true", help="use single timestep readout for online evaluation (only applies to online input 'pred' when dense prediction is used)")
    parser.add_argument("--online_full_spatial_readout", action="store_true", help="use full spatial readout for online evaluation")
    parser.add_argument("--save_online_readout", action="store_true", help="save the online readout weights")
    

def add_offline_eval_args(parser):
    parser.add_argument("--offline_task", choices=["seq2label", "seq2seq", "multitask"], default=None, help="evaluate offline on a classification task")
    parser.add_argument("--offline_input", choices=["enc", "ctx", "pred"], default="enc", help="input to the offline classifier")
    parser.add_argument("--offline_batch_size", type=int, default=256, help="batch size for offline evaluation")
    parser.add_argument("--offline_lr", type=float, default=1e-3, help="learning rate for offline evaluation")
    parser.add_argument("--offline_weight_decay", type=float, default=1e-5, help="weight decay for offline evaluation")
    parser.add_argument("--offline_optimizer", choices=["adam", "sgd"], default="adam", help="optimizer to use for offline evaluation")
    parser.add_argument("--offline_epochs", type=int, default=1, help="number of epochs in offline evaluation")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model checkpoint")
    parser.add_argument("--offline_single_timestep_readout", action="store_true", help="use single timestep readout for online evaluation (only applies to online input 'pred' when dense prediction is used)")
    parser.add_argument("--offline_full_spatial_readout", action="store_true", help="use full spatial readout for offline evaluation")
    parser.add_argument("--save_offline_readout", action="store_true", help="save the offline readout weights")
    parser.add_argument("--use_sklearn_regression", action="store_true", help="use sklearn regression for offline evaluation (only applies to multitask regression tasks)")


def add_ddp_args(parser):
    parser.add_argument("--distributed", action="store_true", help="use distributed training")
    parser.add_argument("--distribute_data", action="store_true", help="distribute data minibatches across nodes")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="distributed backend")
    parser.add_argument("--dist_url", type=str, default="env://", help="distributed url")
    # parser.add_argument("--local_world_size", type=int, default=2, help="number of nodes for distributed training")
    parser.add_argument("--local-rank", type=int, default=0, help="node rank for distributed training")


def check_args(args):
    assert not (args.distributed or args.distribute_data), "Distributed training currently not supported"
    if args.dataset in ["mnist", "oddballs"]:
        if "online_task" in args:
            assert args.online_task == "multitask", "Multitask classification required for MNIST, animals and PFC Oddballs"
        if "offline_task" in args and args.offline_task is not None:
            assert args.offline_task == "multitask", "Multitask classification required for MNIST, animals and PFC Oddballs"
    if args.dataset in ["mnist"]:
        assert args.grayscale, "--grayscale argument required for MNIST dataset"
    if hasattr(args, 'n_areas') and args.n_areas is not None:
        assert args.encoder == "conv2d", "Greedy training is supported for Conv2D encoders"
    if "online_task" in args:
        if args.online_task == "multitask":
            assert args.evaluate_concat_features is False, "Concatenation over time is not supported for multitask classification"
        elif args.online_task == "seq2seq":
            assert args.evaluate_concat_features is False, "Concatenation of features over time cannot be used for seq2seq tasks"
    if "offline_task" in args:
        assert args.dataset != "tlt12", "Please run offline evaluation on TLT12 dataset by following the instructions in the README file."
        if args.offline_task == "multitask":
            assert args.evaluate_concat_features is False, "Concatenation over time is not supported for multitask classification"
        elif args.offline_task == "seq2seq":
            assert args.evaluate_concat_features is False, "Concatenation of features over time cannot be used for seq2seq tasks"
    if "integrator" in args and args.integrator == "identity" and (args.online_input == "ctx" or args.offline_input == "ctx"):
        print("Identity integrator is used. The encoder output will be fed as inputs to readouts.")
    if "loss" in args:
        if args.prediction_target == "pred":
            assert args.loss in ["inv", "ilvic"], "Prediction target cannot be set to pred for losses other than inv and ilvic"
        if args.loss not in ["supervised", "inv", "ilvic"]:
            assert args.predictor is not None, f"{args.loss} loss requires a predictor"
        if args.loss in ["inv", "ilvic"] and args.pred_target_dim_override == 0:
            print("Output size of the predictor network should ideally be specified for inv and ilvic losses using --pred_target_dim_override. If not specified, the output size of the encoder will be used and will be unpredictable.")
    if args.pred_target_dim_override > 0 and "loss" in args:
        assert args.loss in ["inv", "ilvic"], "Output size of the predictor network can be overriden only for inv and ilvic losses using --pred_target_dim_override"
    if args.flatten_images:
        assert args.dataset in ["mnist", "animals", "oddballs"], "Flattening images is only supported for MNIST, Animals and PFC Oddballs datasets"
    print("num_sequences is only used for MNIST datasets. For other datasets, it will be ignored.")
    print("pred_loss_type and pred_target are only used for Pred loss. For other losses, they will be ignored.")
