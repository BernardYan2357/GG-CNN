#!/usr/bin/env python3
"""
Generate a simple URDF from an OBJ mesh.

Example:
python obj_to_urdf.py --obj ./meshes/cube.obj --out ./meshes/cube.urdf --mass 0.2 --scale 1 1 1
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate URDF from OBJ mesh")
    parser.add_argument("--obj", required=True, help="Path to input OBJ file")
    parser.add_argument("--out", required=True, help="Path to output URDF file")
    parser.add_argument(
        "--collision-obj",
        default=None,
        help="Optional collision mesh OBJ path. If omitted, try <obj_basename>_col.obj",
    )
    parser.add_argument("--name", default="obj_model", help="Robot/link base name")
    parser.add_argument("--mass", type=float, default=0.1, help="Link mass in kg")
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Mesh scale: one value (uniform) or three values (sx sy sz)",
    )
    parser.add_argument(
        "--rgba",
        type=float,
        nargs=4,
        default=[0.7, 0.7, 0.7, 1.0],
        help="Material color RGBA",
    )
    parser.add_argument(
        "--keep-origin",
        action="store_true",
        help="Do not recenter mesh; keep OBJ origin as link origin",
    )
    return parser.parse_args()


def normalize_scale(scale_values: list[float]) -> tuple[float, float, float]:
    if len(scale_values) == 1:
        s = float(scale_values[0])
        return s, s, s
    if len(scale_values) == 3:
        return float(scale_values[0]), float(scale_values[1]), float(scale_values[2])
    raise ValueError("--scale must contain 1 or 3 float values")


def parse_obj_bounds(obj_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            min_xyz[0] = min(min_xyz[0], x)
            min_xyz[1] = min(min_xyz[1], y)
            min_xyz[2] = min(min_xyz[2], z)
            max_xyz[0] = max(max_xyz[0], x)
            max_xyz[1] = max(max_xyz[1], y)
            max_xyz[2] = max(max_xyz[2], z)

    if math.isinf(min_xyz[0]):
        raise ValueError(f"No vertex lines found in OBJ: {obj_path}")

    return (min_xyz[0], min_xyz[1], min_xyz[2]), (max_xyz[0], max_xyz[1], max_xyz[2])


def compute_box_inertia(mass: float, size_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size_xyz
    # Prevent zero inertia for flat or tiny meshes.
    eps = 1e-6
    x = max(abs(x), eps)
    y = max(abs(y), eps)
    z = max(abs(z), eps)

    ixx = (mass / 12.0) * (y * y + z * z)
    iyy = (mass / 12.0) * (x * x + z * z)
    izz = (mass / 12.0) * (x * x + y * y)
    return ixx, iyy, izz


def format_vec3(v: tuple[float, float, float]) -> str:
    return f"{v[0]:.8g} {v[1]:.8g} {v[2]:.8g}"


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def build_urdf(
    obj_path: Path,
    collision_obj_path: Path,
    urdf_path: Path,
    name: str,
    mass: float,
    scale: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    keep_origin: bool,
) -> None:
    (min_x, min_y, min_z), (max_x, max_y, max_z) = parse_obj_bounds(obj_path)

    center = (
        ((min_x + max_x) * 0.5) * scale[0],
        ((min_y + max_y) * 0.5) * scale[1],
        ((min_z + max_z) * 0.5) * scale[2],
    )
    size = (
        (max_x - min_x) * scale[0],
        (max_y - min_y) * scale[1],
        (max_z - min_z) * scale[2],
    )

    ixx, iyy, izz = compute_box_inertia(mass, size)

    # Use relative mesh path so moving the folder is easier.
    mesh_rel = os.path.relpath(obj_path.resolve(), urdf_path.parent.resolve()).replace("\\", "/")
    collision_mesh_rel = os.path.relpath(collision_obj_path.resolve(), urdf_path.parent.resolve()).replace("\\", "/")

    robot = ET.Element("robot", attrib={"name": name})
    link = ET.SubElement(robot, "link", attrib={"name": "base_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", attrib={"value": f"{mass:.8g}"})
    ET.SubElement(
        inertial,
        "inertia",
        attrib={
            "ixx": f"{ixx:.8g}",
            "iyy": f"{iyy:.8g}",
            "izz": f"{izz:.8g}",
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )

    mesh_origin = (0.0, 0.0, 0.0) if keep_origin else (-center[0], -center[1], -center[2])

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", attrib={"xyz": format_vec3(mesh_origin), "rpy": "0 0 0"})
    visual_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(
        visual_geom,
        "mesh",
        attrib={
            "filename": mesh_rel,
            "scale": format_vec3(scale),
        },
    )
    material = ET.SubElement(visual, "material", attrib={"name": "mesh_color"})
    ET.SubElement(material, "color", attrib={"rgba": f"{rgba[0]:.6g} {rgba[1]:.6g} {rgba[2]:.6g} {rgba[3]:.6g}"})

    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", attrib={"xyz": format_vec3(mesh_origin), "rpy": "0 0 0"})
    collision_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(
        collision_geom,
        "mesh",
        attrib={
            "filename": collision_mesh_rel,
            "scale": format_vec3(scale),
        },
    )

    indent_xml(robot)
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    args = parse_args()

    obj_path = Path(args.obj).expanduser().resolve()
    urdf_path = Path(args.out).expanduser().resolve()

    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    if args.collision_obj:
        collision_obj_path = Path(args.collision_obj).expanduser().resolve()
    else:
        collision_obj_path = obj_path.with_name(f"{obj_path.stem}_col.obj")
        if not collision_obj_path.exists():
            collision_obj_path = obj_path

    if not collision_obj_path.exists():
        raise FileNotFoundError(f"Collision OBJ file not found: {collision_obj_path}")

    if args.mass <= 0:
        raise ValueError("--mass must be > 0")

    scale = normalize_scale(args.scale)
    rgba = (float(args.rgba[0]), float(args.rgba[1]), float(args.rgba[2]), float(args.rgba[3]))

    build_urdf(
        obj_path=obj_path,
        collision_obj_path=collision_obj_path,
        urdf_path=urdf_path,
        name=args.name,
        mass=float(args.mass),
        scale=scale,
        rgba=rgba,
        keep_origin=bool(args.keep_origin),
    )

    print(f"URDF generated: {urdf_path}")
    print(f"Mesh reference: {obj_path}")
    print(f"Collision mesh reference: {collision_obj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
