import numpy as np

def bgr_to_gray_numpy(img: np.ndarray) -> np.ndarray:
    """使用 NumPy 依據 BT.601 公式將 BGR 影像轉換為灰階影像"""
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    
    # 浮點數加權運算
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    
    # 四捨五入並轉回 8 位元無號整數 (0~255)
    return np.round(gray).astype(np.uint8)