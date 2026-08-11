
export METRIC="${METRIC:-*}"
export SETNUM="${SETNUM:-*}"
export REFERENCE="${REFERENCE:-*}"

# RESULTS_DIR=output/surprise
RESULTS_DIR=~/mnt/jz/physics-eval/output/surprise
echo "##################################"
echo "# Processing results in $RESULTS_DIR/$METRIC/$SETNUM/$REFERENCE/"
echo "##################################"

for path in $RESULTS_DIR/$METRIC/$SETNUM/$REFERENCE/; do
    python ./plot_fid_stats.py --path "$path" --sorted
done