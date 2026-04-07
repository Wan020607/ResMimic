cd legged_gym/legged_gym/scripts
wandb_run_id=${1}
wandb_checkpoint_iter=${2}
WANDB_ENTITY=${3}

# Run the evaluation script
python play_residual.py --task "g1_hoi" \
               --proj_name "resmimic_suitcase" \
               --teacher_exptid "None" \
               --exptid "suitcase" \
               --wandb_run_id ${wandb_run_id} \
               --checkpoint ${wandb_checkpoint_iter} \
               --wandb_entity "$WANDB_ENTITY" \
               --num_envs 4 \
               --device "cuda:0" \
               # --record_video \