import os
import torch
import numpy as np
import meshio
from pathlib import Path
from dataclasses import dataclass
from typing import Union

from .geometryutil import compute_scale_knn

C0 = 0.28209479177387814


@dataclass
class PlyVertexData:
    points: np.ndarray
    point_data: dict[str, np.ndarray]

    @property
    def names(self) -> tuple[str, ...]:
        return ("x", "y", "z", *self.point_data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "x":
            return self.points[:, 0]
        if key == "y":
            return self.points[:, 1]
        if key == "z":
            return self.points[:, 2]
        return self.point_data[key]


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values.reshape(values.shape[0], -1)


def read_ply_vertex_data(file_path: os.PathLike) -> PlyVertexData:
    mesh = meshio.read(Path(file_path))
    points = np.asarray(mesh.points[:, :3], dtype=np.float32)
    point_data = {name: _as_1d_array(values) for name, values in mesh.point_data.items()}
    return PlyVertexData(points=points, point_data=point_data)


def write_ply_vertex_data(path: os.PathLike, points: np.ndarray, point_data: dict[str, np.ndarray]):
    normalized_point_data = {name: np.asarray(values) for name, values in point_data.items()}
    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=np.float32), cells=[], point_data=normalized_point_data
    )
    mesh.write(Path(path), file_format="ply")


def color_to_sh(color: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return (color - 0.5) / C0


def sh_to_color(sh: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    return sh * C0 + 0.5


def construct_list_of_attributes(splats: torch.nn.ParameterDict) -> list[str]:
    lst = [
        "x",
        "y",
        "z",
    ]

    # All channels except the N DC
    for i in range(splats.colors.shape[-1]):
        lst.append("f_dc_{}".format(i))
    lst.append("opacity")
    for i in range(splats.scales.shape[-1]):
        lst.append("scale_{}".format(i))
    for i in range(splats.quats.shape[-1]):
        lst.append("rot_{}".format(i))
    return lst


def save_ply(splats: torch.nn.ParameterDict, path: os.PathLike):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = splats.means.detach().cpu().numpy()
    f_dc = color_to_sh(splats.colors.detach()).cpu().numpy()
    opacities = splats.opacities.detach().unsqueeze(-1).cpu().numpy()
    scale = splats.scales.detach().cpu().numpy()
    rotation = splats.quats.detach().cpu().numpy()
    # normalize quats
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)

    attribute_names = construct_list_of_attributes(splats)
    attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
    point_data = {
        attribute_name: attributes[:, idx].astype(np.float32)
        for idx, attribute_name in enumerate(attribute_names)
        if attribute_name not in {"x", "y", "z"}
    }
    write_ply_vertex_data(path, xyz, point_data)


def load_ply(file_path: str) -> torch.nn.ParameterDict:
    """
    Load Gaussian splat data from a PLY file.

    Args:
        file_path: Path to the PLY file

    Returns:
        ParameterDict containing the loaded splat parameters
    """
    vertex = read_ply_vertex_data(file_path)

    # Extract basic properties
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    attrib_names = vertex.names

    if "opacity" in attrib_names:
        opacities = vertex["opacity"]
    else:
        opacities = np.ones(means.shape[0])

    # Extract scales
    scale_names = [name for name in attrib_names if name.startswith("scale_")]
    if len(scale_names) == 0:
        # run KNN to estimate scales
        scales = compute_scale_knn(means)
    else:
        scales = np.stack([vertex[name] for name in sorted(scale_names)], axis=1)

    # Extract rotations
    rot_names = [name for name in attrib_names if name.startswith("rot_")]
    if len(rot_names) == 0:
        quats = np.zeros((means.shape[0], 4))
        quats[:, 0] = 1.0  # Identity quaternion
    else:
        quats = np.stack([vertex[name] for name in sorted(rot_names)], axis=1)

    # Extract spherical harmonics
    dc_names = [name for name in attrib_names if name.startswith("f_dc_")]
    rest_names = [name for name in attrib_names if name.startswith("f_rest_")]

    if len(dc_names) == 0:
        sh0 = np.zeros((means.shape[0], 3))
    else:
        sh0 = np.stack([vertex[name] for name in sorted(dc_names)], axis=1)

    if rest_names:
        shN = np.stack([vertex[name] for name in sorted(rest_names)], axis=1)
    else:
        shN = np.zeros((means.shape[0], 0))

    # Reshape SH coefficients to match expected format
    sh0 = sh0.reshape(means.shape[0], 1, -1).transpose(0, 2, 1)
    if shN.shape[1] > 0:
        # Assuming shN contains (l-1)^2 coefficients per degree l > 0
        shN.shape[1]
        shN = shN.reshape(means.shape[0], 1, -1).transpose(0, 2, 1)
    else:
        shN = shN.reshape(means.shape[0], 0, 1)

    # Convert to tensors and create ParameterDict
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(means).float()),
            "scales": torch.nn.Parameter(torch.from_numpy(scales).float()),
            "quats": torch.nn.Parameter(torch.from_numpy(quats).float()),
            "opacities": torch.nn.Parameter(torch.from_numpy(opacities).float()),
            "sh0": torch.nn.Parameter(torch.from_numpy(sh0).float()),
            "shN": torch.nn.Parameter(torch.from_numpy(shN).float()),
        }
    )

    return splats
