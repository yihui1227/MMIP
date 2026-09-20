# -*- coding: utf-8 -*-
"""
perspective_utils.py
Quiz 3：梯形校正與透視轉換 — 共用函式模組

放在 Code/ 底下，供 主程式.ipynb import 使用，
讓 notebook 本身保持乾淨、聚焦於流程展示與討論。
"""

from pathlib import Path
import cv2
import numpy as np


# ----------------------------------------------------------------------
# 基礎工具：座標排序 / 四點透視轉換
# ----------------------------------------------------------------------

def order_points(pts):
    """
    將任意順序輸入的 4 個角點，排序成
    [左上, 右上, 右下, 左下] 的順序，這是 cv2.getPerspectiveTransform
    所要求的固定順序。
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    """
    給定原圖與四個角點座標(任意順序)，回傳校正後的正視影像與轉換矩陣 M。
    輸出影像的寬高，是依照四個角點在原圖中對應的實際邊長估計出來的，
    盡量保持校正後物體的長寬比例。
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    max_width = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_height = max(int(heightA), int(heightB))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped, M


# ----------------------------------------------------------------------
# 自動偵測四邊形角點（讓流程可以「自動」跑完多張影像）
# ----------------------------------------------------------------------

def _find_quad_by_contour(edged, resized_shape, min_area_ratio=0.08):
    """在邊緣影像中找面積最大、且可以被近似成 4 個頂點的輪廓。"""
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    img_area = resized_shape[0] * resized_shape[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        contour_area = cv2.contourArea(c)
        if contour_area <= min_area_ratio * img_area:
            continue

        for epsilon_ratio in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(c, epsilon_ratio * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            points = approx.reshape(4, 2)
            if np.any(points[:, 0] <= 2) or np.any(points[:, 1] <= 2):
                continue
            if np.any(points[:, 0] >= resized_shape[1] - 3):
                continue
            if np.any(points[:, 1] >= resized_shape[0] - 3):
                continue
            if cv2.contourArea(approx) > min_area_ratio * img_area:
                return points
    return None


def _find_quad_by_minarearect(edged, resized_shape, min_area_ratio=0.08):
    """
    Stage 2 備援：找不到規則四邊形輪廓時，退而求其次，
    用最大輪廓的最小外接旋轉矩形(minAreaRect)近似。
    這只是粗略近似，對嚴重梯形變形的物體不夠準確，但可以避免流程完全中斷。
    """
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    img_area = resized_shape[0] * resized_shape[1]
    if cv2.contourArea(c) < min_area_ratio * img_area:
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return box


def auto_detect_quad(image, min_area_ratio=0.08, canny_low=50, canny_high=150,
                      return_debug=False):
    """
    嘗試自動找出畫面中要校正的四邊形（例如一張紙、白板、卡片、招牌...）的四個角點。

    流程：
      1) resize 到固定高度(加速、穩定 Canny 參數)
      2) 灰階 + 高斯模糊 + Canny 邊緣偵測
      3) 找輪廓 -> 多邊形近似 -> 挑出近似成 4 個頂點且面積夠大的輪廓
      4) 若找不到規則四邊形，退回用 minAreaRect 粗略近似 (自動化 fallback)

    回傳：
      pts        : 對應回「原圖座標系」的 4 個角點 (float32)，找不到則為 None
      debug_info : dict，包含中介影像與使用的方法，方便除錯 / 報告用
    """
    orig_h = image.shape[0]
    ratio = orig_h / 500.0
    ratio = ratio if ratio > 0 else 1.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    quad = _find_quad_by_contour(edged, resized.shape, min_area_ratio)
    method = "contour_4pt"

    if quad is None:
        quad = _find_quad_by_minarearect(edged, resized.shape, min_area_ratio)
        method = "minAreaRect_fallback"

    if quad is None:
        if return_debug:
            return None, {"method": "failed", "edged": edged, "resized": resized}
        return None

    pts = (quad.astype("float32")) * ratio

    if return_debug:
        return pts, {"method": method, "edged": edged, "resized": resized}
    return pts


# ----------------------------------------------------------------------
# 手動點選角點（自動偵測失敗時的備援方案）
# ----------------------------------------------------------------------

def manual_select_points(image, title="請依序點選 4 個角點（順序不拘）"):
    """
    使用 OpenCV 視窗讓使用者手動點選 4 個角點。
    視窗可能會縮放顯示，但回傳的座標仍會換算回原圖尺寸。
    """
    window_name = "Perspective point selection"
    height, width = image.shape[:2]
    display_scale = min(1200 / width, 800 / height, 1.0)
    display_width = max(int(width * display_scale), 1)
    display_height = max(int(height * display_scale), 1)
    display = cv2.resize(image, (display_width, display_height))
    points = []

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return

        original_point = (x / display_scale, y / display_scale)
        points.append(original_point)
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        if len(points) > 1:
            previous = points[-2]
            cv2.line(
                display,
                (int(previous[0] * display_scale), int(previous[1] * display_scale)),
                (x, y),
                (0, 0, 255),
                2,
            )
        if len(points) == 4:
            first = points[0]
            cv2.line(
                display,
                (int(first[0] * display_scale), int(first[1] * display_scale)),
                (x, y),
                (0, 0, 255),
                2,
            )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, on_mouse)

    cv2.imshow(window_name, display)
    cv2.waitKey(1)
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            ctypes.windll.user32.SetWindowTextW(hwnd, title)
    except AttributeError:
        cv2.setWindowTitle(window_name, title)

    while len(points) < 4:
        cv2.imshow(window_name, display)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.imshow(window_name, display)
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)

    if len(points) != 4:
        raise ValueError("必須點選 4 個角點；按 Esc 取消。")

    return np.array(points, dtype="float32")


# ----------------------------------------------------------------------
# 高階流程：單張 / 批次自動校正
# ----------------------------------------------------------------------

def rectify_image(image_path, out_dir=None, min_area_ratio=0.08,
                   manual_fallback=False, save=True):
    """
    對單一影像執行「自動偵測角點 -> 透視校正」的完整流程。

    manual_fallback=False 時，若自動偵測失敗就直接回傳失敗狀態，
    不會跳出互動視窗中斷批次流程（這是「自動化」的關鍵設計）。
    manual_fallback=True 只建議在單張示範、需要人工協助時開啟。
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return {"path": image_path, "status": "read_error", "warped": None,
                "pts": None, "method": None}

    pts, debug = auto_detect_quad(image, min_area_ratio=min_area_ratio, return_debug=True)
    method = debug["method"]

    if pts is None:
        if manual_fallback:
            pts = manual_select_points(image)
            method = "manual"
        else:
            return {"path": image_path, "status": "detect_failed", "warped": None,
                    "pts": None, "method": method, "debug": debug}

    warped, M = four_point_transform(image, pts)

    result = {"path": image_path, "status": "ok", "warped": warped,
              "pts": pts, "method": method, "debug": debug, "M": M}

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_rectified.jpg"
        cv2.imwrite(str(out_path), warped)
        result["out_path"] = out_path

    return result


def batch_rectify(image_paths, out_dir, min_area_ratio=0.08):
    """
    對多張影像自動套用同一套流程，回傳每張影像的處理結果列表。
    這是「進階：自動化」要求的核心函式 —— 不需人工逐張介入即可跑完全部影像。
    """
    results = []
    for p in image_paths:
        res = rectify_image(p, out_dir=out_dir, min_area_ratio=min_area_ratio,
                             manual_fallback=False, save=True)
        results.append(res)
    return results


def quad_shape_metrics(pts):
    """
    給定偵測到的四邊形角點，計算一些簡單的幾何指標，
    用來輔助判斷「這次自動校正的幾何合理性」，
    可作為角度/條件分析時的量化依據之一。

    回傳：
      edge_lengths      : 四邊長
      edge_ratio        : 最長邊 / 最短邊 (越接近 1 越接近矩形被拍成的規則梯形)
      diag_ratio         : 兩對角線長度比 (越接近 1 通常代表越接近正視角度)
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    edges = [
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - tr),
        np.linalg.norm(bl - br),
        np.linalg.norm(tl - bl),
    ]
    edge_ratio = max(edges) / max(min(edges), 1e-6)

    diag1 = np.linalg.norm(br - tl)
    diag2 = np.linalg.norm(bl - tr)
    diag_ratio = max(diag1, diag2) / max(min(diag1, diag2), 1e-6)

    return {
        "edge_lengths": edges,
        "edge_ratio": edge_ratio,
        "diag_ratio": diag_ratio,
    }