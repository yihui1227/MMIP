from pathlib import Path

import cv2
import numpy as np


def match_sift(image1, image2, ratio=0.75, use_clahe=False):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        return None, keypoints1, keypoints2, []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(descriptors2, descriptors1, k=2)
    good_matches = [
        match for match, second in knn_matches
        if match.distance < ratio * second.distance
    ]

    if len(good_matches) < 4:
        return None, keypoints1, keypoints2, good_matches

    source_points = np.float32([
        keypoints2[match.queryIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    destination_points = np.float32([
        keypoints1[match.trainIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )

    if homography is None or inlier_mask is None:
        return None, keypoints1, keypoints2, good_matches

    return (homography, inlier_mask.ravel().astype(bool)), keypoints1, keypoints2, good_matches


def stitch_images(image1, image2, ratio=0.75, use_clahe=False):
    match_result, keypoints1, keypoints2, good_matches = match_sift(
        image1, image2, ratio=ratio, use_clahe=use_clahe
    )
    if match_result is None:
        return None, {
            'keypoints1': len(keypoints1),
            'keypoints2': len(keypoints2),
            'matches': len(good_matches),
            'inliers': 0,
            'inlier_ratio': 0.0,
            'homography': None,
        }

    homography, inlier_mask = match_result
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    corners2 = np.float32([
        [0, 0], [width2, 0], [width2, height2], [0, height2]
    ]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, homography).reshape(-1, 2)
    corners1 = np.float32([
        [0, 0], [width1, 0], [width1, height1], [0, height1]
    ])
    all_corners = np.vstack([corners1, warped_corners2])
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1],
    ], dtype=np.float64)

    canvas_size = (max_x - min_x, max_y - min_y)
    transform = translation @ homography
    warped2 = cv2.warpPerspective(image2, transform, canvas_size)
    result = warped2.copy()
    first_y = -min_y
    first_x = -min_x
    result[first_y:first_y + height1, first_x:first_x + width1] = image1

    mask1 = np.zeros(canvas_size[::-1], dtype=np.uint8)
    mask1[first_y:first_y + height1, first_x:first_x + width1] = 255
    mask2 = cv2.warpPerspective(
        np.full(image2.shape[:2], 255, dtype=np.uint8),
        transform,
        canvas_size,
    )
    overlap = (mask1 > 0) & (mask2 > 0)
    result[overlap] = (
        0.5 * result[overlap].astype(np.float32)
        + 0.5 * warped2[overlap].astype(np.float32)
    ).astype(np.uint8)

    inlier_count = int(inlier_mask.sum())
    return result, {
        'keypoints1': len(keypoints1),
        'keypoints2': len(keypoints2),
        'matches': len(good_matches),
        'inliers': inlier_count,
        'inlier_ratio': inlier_count / len(good_matches),
        'homography': homography,
    }


def change_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


def reduce_overlap(image, keep_ratio):
    height, width = image.shape[:2]
    crop_width = max(int(width * keep_ratio), 1)
    return image[:, :crop_width]


def run_experiments(image1, image2, ratio=0.75):
    experiments = [
        ('baseline', image2, False),
        ('brighter +60', change_brightness(image2, 60), False),
        ('darker -60', change_brightness(image2, -60), False),
        ('rotated 8 degrees', rotate_image(image2, 8), False),
        ('rotated 20 degrees', rotate_image(image2, 20), False),
        ('reduced overlap', reduce_overlap(image2, 0.55), False),
        ('CLAHE preprocessing', image2, True),
    ]

    rows = []
    results = []
    for name, test_image, use_clahe in experiments:
        stitched, info = stitch_images(
            image1,
            test_image,
            ratio=ratio,
            use_clahe=use_clahe,
        )
        success = (
            stitched is not None
            and info['matches'] >= 12
            and info['inlier_ratio'] >= 0.25
        )
        rows.append({
            'name': name,
            'matches': info['matches'],
            'inliers': info['inliers'],
            'inlier_ratio': info['inlier_ratio'],
            'success': success,
        })
        if stitched is not None:
            results.append((name, stitched, success))

    return rows, results


def load_quiz4_images(data_dir='Data'):
    data_dir = Path(data_dir)
    image1 = cv2.imread(str(data_dir / 'q4-1.jpg'))
    image2 = cv2.imread(str(data_dir / 'q4-2.jpg'))
    if image1 is None or image2 is None:
        raise FileNotFoundError('找不到 q4-1.jpg 或 q4-2.jpg')
    return image1, image2
