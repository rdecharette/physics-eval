# Vendored dependencies

`clean-fid` and `WMReward` are tracked directly in physics-eval, including local
changes. Edit and commit their files from the physics-eval repository as usual.

`WMReward/vjepa2` and `WMReward/MAGI-1` remain Git submodules. Initialize them
from the physics-eval repository root with:

```sh
git submodule update --init --recursive
```

Changes within those two submodules must be committed and pushed in their own
repositories, then their updated revisions committed in physics-eval.

The source revisions at the time of conversion are recorded below; local
modifications may differ from these revisions. Original licenses are retained.

| Directory | Source | Revision |
| --- | --- | --- |
| `third_party/clean-fid` | https://github.com/GaParmar/clean-fid.git | `e88c4d6269a4bbf04c04deeb578475b57719acee` |
| `third_party/WMReward` | git@github.com:rdecharette/WMReward.git | `a539e8dfb46f6067dac614ee6c8a551b4fd540ac` |
| `third_party/WMReward/vjepa2` | https://github.com/facebookresearch/vjepa2.git | `c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb` |
| `third_party/WMReward/MAGI-1` | https://github.com/YuanJianhao508/MAGI-1.git | `27e0a2ebc376039a548bb257dbef12e7c94bec02` |
