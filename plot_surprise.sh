export MODEL="${MODEL:-*}"
export RUN="${RUN:-*}"
export METHOD="${METHOD:-*}"

# RESULTS_DIR=output/surprise
RESULTS_DIR=~/mnt/jz/physics-eval/output/surprise
echo "##################################"
echo "# Processing results in $RESULTS_DIR/$MODEL/$METHOD/$RUN/"
echo "##################################"

for path in $RESULTS_DIR/$MODEL/$METHOD/$RUN/; do
    echo -e "\n\n# Processing $path"
    python ./plot_surprise_stats.py --path "$path" --sorted
 done