#!/usr/bin/env bash
set -euo pipefail

# Usage from launcher:
#   JOB_PROFILE=<name> bash scripts/surprise/job_all.sh
#
# Profiles set JOB_LAUNCHER, JOB_PROFILE_ARGS, and optional JOB_HEADER commands.
JOB_TIME=${JOB_TIME:-}

job_profile_set_jeanzay_common() {
    JOB_LAUNCHER="sbatch"
    JOB_PROFILE_ARGS=(
        --account=pab@a100
        --constraint=a100
        --nodes=1
        --ntasks-per-node=1
        --cpus-per-task=16
        --gres=gpu:1
        --hint=nomultithread
    )
    JOB_HEADER=(
        # Set all caches (probably not needed since we have done the proper symlink)
        "export TORCH_HOME=\$SCRATCH/.cache/torch"
        "mkdir -p \$TORCH_HOME"

        # Load the required modules for the job
        "module load arch/a100"
        "module load anaconda-py3/2024.06"
        "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
        "conda activate ~/.conda/envs/physics-eval/"

        # Ensure the dataset is untared on SCRATCH where we have tons of inodes/storage
        "if [[ -n \"\${REQUIRED_DATASET:-}\" ]]; then bash \"$ROOT_DIR/jz_assess_dataset.sh\" \"\$REQUIRED_DATASET\"; fi"
    )
}

job_profile_config() {
    local profile="$1"

    case "$profile" in
        # JEANZAY cluster A100 dev partition
        jeanzay-a100-dev)
            job_profile_set_jeanzay_common
            if [[ -z "$JOB_TIME" ]]; then
                JOB_TIME="02:00:00"
            fi
            JOB_PROFILE_ARGS+=(
                --qos=qos_gpu_a100-dev
                --time="$JOB_TIME"
            )
            ;;

        # JEANZAY cluster A100 t3 partition
        jeanzay-a100-t3)
            job_profile_set_jeanzay_common
            if [[ -z "$JOB_TIME" ]]; then
                JOB_TIME="10:00:00"
            fi
            JOB_PROFILE_ARGS+=(
                --qos=qos_gpu_a100-t3
                --time="$JOB_TIME"
            )
            ;;

        # SALSA cluster
        salsa)
            JOB_LAUNCHER="sbatch"
            JOB_PROFILE_ARGS=(
                --partition=debug
                --nodes=1
                --ntasks-per-node=1
                --cpus-per-task=8
                --gres=gpu:1
                --mem=24G
            )
            JOB_HEADER=(
                "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
                "conda activate physics-eval"
            )
            ;;

        *)
            echo "Unknown JOB_PROFILE '$profile'" >&2
            echo "Supported profiles: jeanzay-a100-dev, jeanzay-a100-t3, salsa" >&2
            return 1
            ;;
    esac
}
