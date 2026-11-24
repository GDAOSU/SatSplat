import torch
import numpy as np
from typing import Union


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    Args:
        quat: [N, 4] quaternions in (w, x, y, z) format
    Returns:
        [N, 3, 3] rotation matrices
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Normalize quaternions
    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Build rotation matrix
    R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)

    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w

    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w

    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y

    return R


def rotation_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    Args:
        R: [N, 3, 3] rotation matrices
    Returns:
        [N, 4] quaternions in (w, x, y, z) format
    """
    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype

    # Allocate quaternion tensor
    quat = torch.zeros(batch_size, 4, device=device, dtype=dtype)

    # Compute trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        quat[mask1, 0] = 0.25 * s  # qw
        quat[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
        quat[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
        quat[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4 * qx
        quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
        quat[mask2, 1] = 0.25 * s  # qx
        quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
        quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4 * qy
        quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
        quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
        quat[mask3, 2] = 0.25 * s  # qy
        quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz

    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4 * qz
        quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
        quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
        quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
        quat[mask4, 3] = 0.25 * s  # qz

    return quat


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return quaternion rotating v_from onto v_to (wxyz order)."""
    v0 = np.asarray(v_from, dtype=np.float32)
    v1 = np.asarray(v_to, dtype=np.float32)

    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)
    if norm0 < 1e-8 or norm1 < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    v0 /= norm0
    v1 /= norm1

    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))

    if dot < -0.999999:
        # 180-degree rotation: choose an orthogonal axis
        orthogonal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(v0[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(v0, orthogonal)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            axis_norm = np.linalg.norm(axis)
        axis /= axis_norm
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)

    axis = np.cross(v0, v1)
    quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float32)
    quat /= np.linalg.norm(quat) + 1e-8
    return quat


def compute_scale_knn(points: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Estimate splat scales from the RMS distance to three neighboring centers."""
    from scipy.spatial import cKDTree

    is_tensor = isinstance(points, torch.Tensor)
    centers = points.detach().cpu().numpy() if is_tensor else points
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 2:
        raise ValueError("Expected at least two 3D centers.")

    distances, _ = cKDTree(centers).query(centers, k=min(4, len(centers)))
    mean_squared = np.square(distances[:, 1:]).mean(axis=1)
    scales = np.repeat(np.sqrt(np.maximum(mean_squared, 1e-7))[:, None], 3, axis=1)
    if is_tensor:
        return torch.as_tensor(scales, dtype=points.dtype, device=points.device)
    return scales.astype(centers.dtype, copy=False)
