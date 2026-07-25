export MODEL="${MODEL:-*}"
export RUN="${RUN:-*}"
export METHOD="${METHOD:-*}"

for path in output/surprise/$MODEL/$METHOD/$RUN/; do
    python ./plot_surprise_stats.py --path "$path" --sorted
 done