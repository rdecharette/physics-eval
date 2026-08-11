# export RUN="${RUN:-*}"
export FORMAT="${FORMAT:-*}"
# args_missing=" --missing-dimension"
args_missing=""

# RESULTS_DIR=output/vbench
RESULTS_DIR=~/mnt/jz/physics-eval/output/vbench
echo "##################################"
echo "# Processing results in $RESULTS_DIR/$FORMAT/"
echo "##################################"

for path in $RESULTS_DIR/$FORMAT/; do
    echo -e "\n# Processing $path"
    python ./plot_vbench_stats.py --path "$path" --missing-dimension
    python ./plot_vbench_stats.py --path "$path" --ignore-dimension dynamic_degree
    python ./plot_vbench_stats.py --path "$path"
done