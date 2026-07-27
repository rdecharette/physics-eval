configs=(
    "48 24 8 topp50 mean max"
    "48 24 8 topp50 mean max"
    # "16 8 8 topp50 mean"
    # "16 8 4 topp50 mean"

    # # "48 24 4 topp50 mean"
    # "48 24 2 topp50 mean"
)

for config in "${configs[@]}"; do
    set -- $config
    export WINDOW_SIZE="$1"
    export CONTEXT_FRAMES="$2"
    export STRIDE="$3"
    shift 3

    for mode in "$@"; do
        MODE="$mode" bash ./scripts/surprise/job_all.sh
    done
done