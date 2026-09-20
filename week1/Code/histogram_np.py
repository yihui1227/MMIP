import numpy as np

def equalize_hist_numpy(gray_img: np.ndarray) -> np.ndarray:
    """
    使用 NumPy 自行實作的灰階直方圖均衡化
    輸入必須為單通道 uint8 灰階影像
    """
    # 計算每個灰階值 0~255 出現次數
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=[0, 256])
    
    # 計算累積分布函數 (CDF)
    cdf = hist.cumsum()
    
    # 排除次數為 0 的像素，並使用正規化公式映射到 0~255
    cdf_mask = np.ma.masked_equal(cdf, 0)
    cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
    cdf_final = np.ma.filled(cdf_mask, 0).astype('uint8')
    
    return cdf_final[gray_img]