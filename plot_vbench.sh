# export RUN="${RUN:-*}"
export FORMAT="${FORMAT:-*}"
# args_missing=" --missing-dimension"
args_missing=""

for path in output/vbench/$FORMAT/; do
    python ./plot_vbench_stats.py --path "$path" --missing-dimension
    python ./plot_vbench_stats.py --path "$path"
 done