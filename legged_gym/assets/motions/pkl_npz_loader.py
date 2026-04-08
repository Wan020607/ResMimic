import sys
import pickle
import numpy as np
import torch
from types import ModuleType


# =========================
# 1. 解决 numpy._core 报错（关键）
# =========================
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)


# patch numpy 内部模块路径
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core)
sys.modules['numpy._core.multiarray'] = FakeModule(
    'numpy._core.multiarray', getattr(np.core, 'multiarray', None)
)
sys.modules['numpy._core.numeric'] = FakeModule(
    'numpy._core.numeric', getattr(np.core, 'numeric', None)
)
sys.modules['numpy._core.fromnumeric'] = FakeModule(
    'numpy._core.fromnumeric', getattr(np.core, 'fromnumeric', None)
)
sys.modules['numpy._core.umath'] = FakeModule(
    'numpy._core.umath', getattr(np.core, 'umath', None)
)

# 兼容旧路径
sys.modules['numpy.core.multiarray'] = np.core.multiarray
sys.modules['numpy.core._multiarray_umath'] = np.core._multiarray_umath


# =========================
# 2. 打印函数（按顺序递归）
# =========================
def print_structure(data, indent=0, name="root"):
    prefix = " " * indent

    # dict（按顺序）
    if isinstance(data, dict):
        print(f"{prefix}{name} -> dict (len={len(data)})")
        for key in data:  # 保持原顺序
            print_structure(data[key], indent + 4, name=key)

    # list / tuple
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{name} -> {type(data).__name__} (len={len(data)})")

        # ====== 特殊展开 body names ======
        if name in ["link_body_list", "body_names"] and len(data) > 0:
            for i, item in enumerate(data):
                print(f"{prefix}    [{i}] {item}")

        elif len(data) > 0 and isinstance(data[0], str):
            for i, item in enumerate(data):
                print(f"{prefix}    [{i}] {item}")

        elif len(data) > 0:
            print_structure(data[0], indent + 4, name=f"{name}[0]")

    # numpy array
    elif isinstance(data, np.ndarray):
        print(f"{prefix}{name} -> ndarray, shape={data.shape}, dtype={data.dtype}")

    # torch tensor
    elif torch.is_tensor(data):
        print(f"{prefix}{name} -> tensor, shape={tuple(data.shape)}, dtype={data.dtype}")

    # 基本类型
    elif isinstance(data, (int, float, str, bool)):
        print(f"{prefix}{name} -> {type(data).__name__}, value={data}")

    # 自定义对象
    elif hasattr(data, "__dict__"):
        print(f"{prefix}{name} -> object: {type(data)}")
        for key, value in data.__dict__.items():
            print_structure(value, indent + 4, name=key)

    else:
        print(f"{prefix}{name} -> {type(data)}")


# =========================
# 3. 主函数
# =========================
def load_and_inspect(file_path):
    try:
        # ====== 根据后缀判断 ======
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            print("[INFO] 成功加载 PKL 文件\n")

        elif file_path.endswith(".npz"):
            npz_data = np.load(file_path, allow_pickle=True)
            data = dict(npz_data)  # 转成 dict
            print("[INFO] 成功加载 NPZ 文件\n")

        else:
            raise ValueError("不支持的文件类型")

    except Exception as e:
        print("[ERROR] 加载失败：", e)
        return

    print("========== 数据结构 ==========\n")
    print_structure(data)

    # 针对 pkl 和 npz 文件打印特定数据
    # =========================
    if file_path.endswith(".pkl") and isinstance(data, dict):
        if "root_rot" in data:
            root_rot = data["root_rot"]
            if isinstance(root_rot, (list, np.ndarray, torch.Tensor)) and len(root_rot) > 0:
                print("\n========== PKL: Root Rotation (Frame 0) ==========\n")
                print("root_rot[0] =", root_rot[0])
        else:
            print("未在 PKL 中找到 root_rot")

    elif file_path.endswith(".npz") and isinstance(data, dict):
        if "joint_pos" in data:
            joint_pos = data["joint_pos"]
            # 有些 npz 是 ndarray(object)
            if isinstance(joint_pos, np.ndarray) and joint_pos.dtype == object:
                joint_pos = joint_pos.tolist()
            if len(joint_pos) > 0:
                print("\n========== NPZ: joint_pos (Frame 0) ==========\n")
                print(joint_pos[0])
        else:
            print("未在 NPZ 中找到 joint_pos")

    # if file_path.endswith(".npz") and isinstance(data, dict):
    #     if "joint_names" in data:
    #         joint_names = data["joint_names"]
    #         # 有些 npz 是 ndarray(object)
    #     if isinstance(joint_names, np.ndarray):
    #         joint_names = joint_names.tolist()

    #     print("来自 NPZ: joint_names")
    #     for i, name in enumerate(joint_names):
    #         print(f"{i}: {name}")

    # else:
    #     print("未找到 joint_names 或 link_body_list")

    if file_path.endswith(".npz") and isinstance(data, dict):
        if "qpos" in data:
            qpos = data["qpos"]
            # 有些 npz 是 ndarray(object)
        if isinstance(qpos, np.ndarray):
            qpos = qpos.tolist()
        if len(qpos) > 0:
            print("\n========== NPZ: qpos (Frame 0) ==========\n")
            print(qpos[0][7:36])

    # ====== 统一打印 body names ======
    print("\n========== Body Names ==========\n")

    # pkl 格式
    if isinstance(data, dict) and "link_body_list" in data:
        print("来自 PKL: link_body_list")
        for i, name in enumerate(data["link_body_list"]):
            print(f"{i}: {name}")

    # npz 格式
    elif isinstance(data, dict) and "body_names" in data:
        body_names = data["body_names"]

        # 有些 npz 是 ndarray(object)
        if isinstance(body_names, np.ndarray):
            body_names = body_names.tolist()

        print("来自 NPZ: body_names")
        for i, name in enumerate(body_names):
            print(f"{i}: {name}")

    else:
        print("未找到 body_names 或 link_body_list")

# =========================
# 4. 入口
# =========================
if __name__ == "__main__":
    file_path = "1_original.npz"  # 改成你的路径
    load_and_inspect(file_path)