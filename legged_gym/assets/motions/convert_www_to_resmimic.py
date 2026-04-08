import numpy as np
import pickle
from pathlib import Path

# =========================
# Quaternion 工具函数
# =========================
def quat_conjugate(q):
    """q: (..., 4) [x, y, z, w]"""
    q_conj = q.copy()
    q_conj[..., :3] *= -1
    return q_conj

def quat_rotate(q, v):
    """
    q: (..., 4) [x, y, z, w]
    v: (..., 3)
    """
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)

def convert_to_local_root_body_pos(root_pos, root_rot, body_pos):
    """
    root_pos: (T, 3)
    root_rot: (T, 4) xyzw
    body_pos: (T, B, 3)
    """
    # 1. 减去 root 平移
    body_pos_rel = body_pos - root_pos[:, None, :]

    # 2. 逆旋转
    root_inv_rot = quat_conjugate(root_rot)

    # 3. 扩展到所有 body
    root_inv_rot_expand = np.repeat(root_inv_rot[:, None, :], body_pos.shape[1], axis=1)

    # 4. flatten
    flat_q = root_inv_rot_expand.reshape(-1, 4)
    flat_v = body_pos_rel.reshape(-1, 3)

    # 5. 旋转
    flat_local = quat_rotate(flat_q, flat_v)

    # 6. reshape
    local_body_pos = flat_local.reshape(body_pos.shape)

    return local_body_pos

def load_dof_pos_from_original_npz(original_npz_path, target_frames, target_fps):
    """
    从原始 NPZ 中提取 qpos[:, 7:36] 作为 dof_pos。
    如果帧数或 fps 不一致，则按时间轴插值到目标序列。
    """
    original_data = np.load(original_npz_path, allow_pickle=True)

    if "qpos" not in original_data:
        raise KeyError(f"'qpos' not found in {original_npz_path}")

    qpos = np.asarray(original_data["qpos"])
    if qpos.ndim != 2 or qpos.shape[1] < 36:
        raise ValueError(
            f"Expected qpos with shape (T, >=36), but got {qpos.shape} from {original_npz_path}"
        )

    source_dof_pos = qpos[:, 7:36]
    source_frames = source_dof_pos.shape[0]
    source_fps = int(np.asarray(original_data["fps"]).reshape(-1)[0]) if "fps" in original_data else target_fps

    if source_frames == target_frames and source_fps == target_fps:
        return source_dof_pos

    # 用每段 motion 的真实时长做时间对齐，避免 30fps / 50fps 直接按索引错位。
    source_time = np.arange(source_frames, dtype=np.float64) / float(source_fps)
    target_time = np.arange(target_frames, dtype=np.float64) / float(target_fps)
    aligned_dof_pos = np.empty((target_frames, source_dof_pos.shape[1]), dtype=source_dof_pos.dtype)

    for dim in range(source_dof_pos.shape[1]):
        aligned_dof_pos[:, dim] = np.interp(
            target_time,
            source_time,
            source_dof_pos[:, dim],
            left=source_dof_pos[0, dim],
            right=source_dof_pos[-1, dim],
        )

    print(
        f"[INFO] Resampled dof_pos from {source_frames}@{source_fps}fps "
        f"to {target_frames}@{target_fps}fps using qpos[:, 7:36]"
    )
    return aligned_dof_pos

# =========================
# 主函数
# =========================
def convert_npz_to_pkl(npz_path, original_npz_path, out_pkl_path, out_npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # =========================
    # 读取数据
    # =========================
    fps = int(data["fps"][0])
    body_pos = data["body_pos_w"]                      # (T, 30, 3)
    body_quat = data["body_quat_w"]                    # (T, 30, 4) wxyz
    body_names = data["body_names"].tolist()
    object_pos = data["object_pos_w"]                  # (T, 3)
    object_quat = data["object_quat_w"]                # (T, 4) wxyz

    # =========================
    # ⚠️ 转换 wxyz -> xyzw
    # =========================
    body_quat = np.concatenate([body_quat[..., 1:], body_quat[..., :1]], axis=-1)
    object_quat = np.concatenate([object_quat[..., 1:], object_quat[..., :1]], axis=-1)

    # =========================
    # root (pelvis)
    # =========================
    root_pos = body_pos[:, 0, :]       # (T, 3)
    root_rot = body_quat[:, 0, :]      # (T, 4) xyzw

    # =========================
    # 使用向量化版本计算 local_body_pos
    # =========================
    local_body_pos = convert_to_local_root_body_pos(
        root_pos,
        root_rot,
        body_pos
    )

    dof_pos = load_dof_pos_from_original_npz(
        original_npz_path=original_npz_path,
        target_frames=body_pos.shape[0],
        target_fps=fps,
    )

    # =========================
    # 生成 PKL
    # =========================
    pkl_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,          # xyzw
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
    }

    with open(out_pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)

    print(f"[INFO] Saved PKL -> {out_pkl_path}")

    # =========================
    # 生成新的 NPZ
    # =========================
    np.savez(
        out_npz_path,
        trans=object_pos,
        rot=object_quat              # xyzw
    )

    print(f"[INFO] Saved NPZ -> {out_npz_path}")

    # =========================
    # Debug 检查
    # =========================
    print("\n[DEBUG] pelvis local pos (should ~0):")
    print(np.mean(local_body_pos[:, 0, :], axis=0))

# =========================
# 入口
# =========================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    convert_npz_to_pkl(
        base_dir / "sub3_largebox_003_beyondmimic_w_obj.npz",
        base_dir / "1_original.npz",
        base_dir / "www_obj.pkl",
        base_dir / "www_obj.npz",
    )
