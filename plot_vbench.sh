export RUN="${RUN:-*}"
export FORMAT="${FORMAT:-*}"

for path in output/vbench/$FORMAT/$RUN/; do
    python ./plot_vbench_stats.py --path "$path"
 done