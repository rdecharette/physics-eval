# #ContPhy videos
# find datasets/ContPhy/ -type f -name "output_Full.mp4" > contphy.txt

# PisaBench videos (you need to have created the videos first using the download_pisabench.sh script)
find datasets/pisabench/real_sync/ -type f -name "*.mp4" > pisabench.txt

# # PhysicsIQ-verified
# find datasets/physics-iq-verified/full-videos/ -type f -name "*.mp4" > physics-iq-verified.txt

# PhysBench
# find datasets/PhysBench/ -type f -name "*.mp4" > PhysBench.txt


# # Newtphys random videos
# find "datasets/NewtPhys/dl3dv/random/1/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_1.txt
# find "datasets/NewtPhys/dl3dv/random/2/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_2.txt
# find "datasets/NewtPhys/dl3dv/random/3/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_3.txt
# find "datasets/NewtPhys/dl3dv/random/4/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_4.txt
# find "datasets/NewtPhys/dl3dv/random/5/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_5.txt
# find "datasets/NewtPhys/dl3dv/random/6/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_6.txt
# find "datasets/NewtPhys/dl3dv/random/7/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_7.txt
# find "datasets/NewtPhys/dl3dv/random/8/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_8.txt
# find "datasets/NewtPhys/dl3dv/random/9/" \
#   -type d -name '_invalid' -prune -o \
#   -type f -name '_fps-25_render.mp4' -print > newtphys_random_9.txt

# IntPhys2 videos
# python intphys2_split.py

# INTPHYS2_ROOT="datasets/intphys2/Main"
# awk -v root="$INTPHYS2_ROOT" 'NF { print root "/" $0 }' "$INTPHYS2_ROOT/possible.txt" > intphys2_possible.txt
# awk -v root="$INTPHYS2_ROOT" 'NF { print root "/" $0 }' "$INTPHYS2_ROOT/impossible.txt" > intphys2_impossible.txt


# # IntPhys2 videos
# python intphys_split.py

# INTPHYS_ROOT="datasets/intphys/dev"
# awk -v root="$INTPHYS_ROOT" 'NF { print root "/" $0 }' "$INTPHYS_ROOT/possible.txt" > intphys_possible.txt
# awk -v root="$INTPHYS_ROOT" 'NF { print root "/" $0 }' "$INTPHYS_ROOT/impossible.txt" > intphys_impossible.txt

# Physion++ videos
# find datasets/physionpp_trim-e200/data_v1/ -type f -name "*_img.mp4" > physionpp.txt

# SEED="42"
# awk -v seed="$SEED" 'BEGIN{srand(seed)} {print rand() "\t" $0}' physionpp.txt \
#   | sort -k1,1n \
#   | cut -f2- > physionpp.txt.tmp && mv physionpp.txt.tmp physionpp.txt