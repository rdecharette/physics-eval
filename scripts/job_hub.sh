#!/usr/bin/env bash
set -euo pipefail

slurm_write_runner_header() {
    local runner_script="$1"

    cat >"$runner_script" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start: $(date)"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "Effective CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
EOF
}

job_submit() {
    local job_name="$1"
    local slurm_log_dir="$2"
    local chdir_dir="$3"
    local runner_body_fn="$4"

    local root_dir="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local job_profile="${JOB_PROFILE:-}"
    local job_profile_file="${JOB_PROFILE_FILE:-$root_dir/scripts/job_profiles.sh}"

    if [[ -z "$job_profile" ]]; then
        echo "JOB_PROFILE is required (example: JOB_PROFILE=jeanzay-a100-dev)" >&2
        return 1
    fi
    if [[ ! -f "$job_profile_file" ]]; then
        echo "Profile file not found: $job_profile_file" >&2
        return 1
    fi

    # shellcheck source=/dev/null
    source "$job_profile_file"

    if ! declare -F job_profile_config >/dev/null 2>&1; then
        echo "Profile file does not define job_profile_config(): $job_profile_file" >&2
        return 1
    fi

    JOB_LAUNCHER=""
    JOB_PROFILE_ARGS=()
    JOB_HEADER=()
    job_profile_config "$job_profile"

    if [[ -z "${JOB_LAUNCHER:-}" ]]; then
        echo "Profile '$job_profile' did not set JOB_LAUNCHER" >&2
        return 1
    fi

    local -a profile_args=("${JOB_PROFILE_ARGS[@]}")
    local -a job_header=("${JOB_HEADER[@]}")

    mkdir -p "$slurm_log_dir"

    local runner_script="$slurm_log_dir/run_$(date +%Y%m%d_%H%M%S)_${job_name}_$$.sh"
    slurm_write_runner_header "$runner_script"
    if [[ ${#job_header[@]} -gt 0 ]]; then
        {
            printf '\n'
            for cmd in "${job_header[@]}"; do
                printf '%s\n' "$cmd"
            done
            printf '\n'
        } >>"$runner_script"
    fi
    if [[ -n "$runner_body_fn" ]]; then
        "$runner_body_fn" >>"$runner_script"
    fi
    chmod +x "$runner_script"

    local -a common_args=(
        --job-name "$job_name"
        --chdir "$chdir_dir"
        "${profile_args[@]}"
    )

    case "$JOB_LAUNCHER" in
        sbatch)
            local -a sbatch_args=(
                --parsable
                --output "$slurm_log_dir/%j_%x.log"
                --error "$slurm_log_dir/%j_%x.log"
            )
            local job_id
            job_id=$(sbatch "${sbatch_args[@]}" "${common_args[@]}" "$runner_script")
            printf 'Submitted %s as job %s\n' "$job_name" "$job_id"
            ;;
        srun)
            echo "Running interactive debug job for $job_name"
            printf 'Running:\nsrun '
            printf '%q ' "${common_args[@]}" "$runner_script"
            printf '\n'
            srun "${common_args[@]}" "$runner_script"
            ;;
        *)
            echo "Unsupported JOB_LAUNCHER='$JOB_LAUNCHER' (expected: sbatch or srun)" >&2
            return 1
            ;;
    esac
}
