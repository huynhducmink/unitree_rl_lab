#!/bin/bash

pkgs=(numpy h5py open3d torch tensordict bpy matplotlib gymnasium gym scipy hidapi pyyaml prettytable toml trimesh tqdm psutil); \
for p in "${pkgs[@]}"; do python -m pip show "$p" >/dev/null 2>&1 && \
  echo "$p: $(python -m pip show "$p" | awk -F': ' '/^Version:/{print $2}')" || echo "$p: MISSING"; done
