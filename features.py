import numpy as np


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def extract_features(world_landmarks):
    """
    world_landmarks:
        MediaPipe 的 21 个 hand_world_landmarks

    return:
        63维归一化特征
    """

    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in world_landmarks],
        dtype=np.float32
    )

    # -------------------------
    # 1. 平移：wrist 作为原点
    # -------------------------
    wrist = pts[0].copy()
    pts = pts - wrist

    # MediaPipe:
    # 0  = wrist
    # 5  = index MCP
    # 9  = middle MCP
    # 17 = pinky MCP

    # -------------------------
    # 2. 建立手掌局部坐标系
    # -------------------------

    # Y轴：手腕 -> 中指根部
    y_axis = _normalize(pts[9])

    # 初始 X轴：小指根部 -> 食指根部
    x_temp = _normalize(pts[5] - pts[17])

    # Z轴：手掌法向
    z_axis = _normalize(np.cross(x_temp, y_axis))

    # 重新计算 X，保证正交
    x_axis = _normalize(np.cross(y_axis, z_axis))

    # -------------------------
    # 3. 转入手掌坐标系
    # -------------------------
    local = np.stack([
        pts @ x_axis,
        pts @ y_axis,
        pts @ z_axis
    ], axis=1)

    # -------------------------
    # 4. 尺度归一化
    # -------------------------
    palm_size = (
        np.linalg.norm(pts[5]) +
        np.linalg.norm(pts[9]) +
        np.linalg.norm(pts[17])
    ) / 3.0

    if palm_size > 1e-8:
        local /= palm_size

    # 21 × 3 -> 63
    return local.flatten()
