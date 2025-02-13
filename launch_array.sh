
# feedforward 

# parameter sweep for 16 grid
param_sweep () {
    # baseline
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 24

    # vary LR
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 1e-2 --init_weight_scale 1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 1e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 1e-2 --init_weight_scale 1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 1e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 24


    # vary init weight
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 0.1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 10 --num_layers 2 --latent_channels 24

    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 0.1 --num_layers 2 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 10 --num_layers 2 --latent_channels 24


    # vary # layers
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 1 --latent_channels 24
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 3 --latent_channels 24
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 4 --latent_channels 24

    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 1 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 3 --latent_channels 24
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 4 --latent_channels 24

    # vary # latent channels
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 16
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 32

    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 16
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 1 --num_layers 2 --latent_channels 32

    # try super big variant
    sbatch run_train.sh -c configs/ff.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 0.1 --num_layers 4 --latent_channels 48
    sbatch run_train.sh -c configs/therino.json --ds_type fixed16 --lr_max 5e-3 --init_weight_scale 0.1 --num_layers 4 --latent_channels 48
}

sweep_dataset () {
    # sweep all models over dataset using "best" params
    ds=$1

    echo Sweeping dataset $1!

    other_params="--lr_max 5e-3 --init_weight_scale 0.1 --num_layers 2 --latent_channels 24"
    other_params_big="--lr_max 5e-3 --init_weight_scale 0.1 --num_layers 4 --latent_channels 48"

    # # feedforward regular, ema, big
    # sbatch run_train.sh -c configs/ff.json --ds_type $ds $other_params
    # sbatch run_train.sh -c configs/ff.json --ds_type $ds $other_params_big

    # # ifno regular, ema
    # sbatch run_train.sh -c configs/ifno.json --ds_type $ds $other_params

    # # fno-deq regular, ema
    # sbatch run_train.sh -c configs/fno_deq.json --ds_type $ds $other_params

    # # therino regular, ema
    # sbatch run_train.sh -c configs/therino.json --ds_type $ds $other_params

    # # no therm therino (a.k.a. MFD)
    # sbatch run_train.sh -c configs/therino_notherm.json --ds_type $ds $other_params

    sbatch run_train.sh -c configs/therino_pre.json --ds_type $ds $other_params
    sbatch run_train.sh -c configs/therino_post.json --ds_type $ds $other_params
    sbatch run_train.sh -c configs/therino_hybrid.json --ds_type $ds $other_params
    # sbatch run_train.sh -c configs/therino.json --ds_type $ds $other_params


    # all ema as well
    # sbatch run_train.sh -c configs/ff.json --ds_type $ds $other_params -E
    # sbatch run_train.sh -c configs/ff.json --ds_type $ds $other_params_big -E
    # sbatch run_train.sh -c configs/ifno.json --ds_type $ds $other_params -E
    # sbatch run_train.sh -c configs/fno_deq.json --ds_type $ds $other_params -E
    # sbatch run_train.sh -c configs/therino.json --ds_type $ds $other_params -E
    # sbatch run_train.sh -c configs/therino_notherm.json --ds_type $ds $other_params -E
}


param_sweep 

# sweep 32^3 datasets
# sweep_dataset fixed32
# sweep_dataset randbc32
# sweep_dataset randcr32

# # 16^3 datasets
# sweep_dataset fixed16_u2
# sweep_dataset fixed16
