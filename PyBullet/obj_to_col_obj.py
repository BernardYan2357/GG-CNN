#!/usr/bin/env python3
"""
Batch-generate collision meshes (*_col.obj) from OBJ files via PyBullet VHACD.

Usage:
python obj_to_col_obj.py --path D:/your/obj/folder
"""

import argparse
import os

import pybullet as p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate *_col.obj files from OBJ using VHACD")
    parser.add_argument("--path", required=True, help="Folder that contains .obj files")
    parser.add_argument("--log", default="log.txt", help="VHACD log file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = os.path.abspath(args.path)

    if not os.path.isdir(path):
        raise NotADirectoryError(f"Invalid folder path: {path}")

    # 防止重复连接导致连接错误。
    while p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)

    try:
        files = os.listdir(path)
        processed = 0

        for file in files:
            if not file.lower().endswith(".obj"):
                continue
            if file.lower().endswith("_col.obj"):
                continue

            print("processing ...", file)
            name_in = os.path.join(path, file)
            name_out = os.path.join(path, file[:-4] + "_col.obj")
            name_log = os.path.abspath(args.log)

            p.vhacd(name_in, name_out, name_log) # VHACD分解，生成碰撞模型
            processed += 1

        print(f"Done. Processed {processed} OBJ file(s).")
    finally:
        if p.isConnected():
            p.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
