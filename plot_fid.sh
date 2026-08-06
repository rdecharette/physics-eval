
export METRIC="${METRIC:-*}"
export SETNUM="${SETNUM:-*}"
export REFERENCE="${REFERENCE:-*}"

for path in output/cleanfid/$METRIC/$SETNUM/$REFERENCE/; do
    python ./plot_fid_stats.py --path "$path" --sorted
done