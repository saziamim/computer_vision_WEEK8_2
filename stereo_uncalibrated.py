import cv2
import numpy as np
import argparse
import os
import textwrap

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)

def format_matrix(name, M):
    return f"{name} =\n{np.array2string(M, precision=6, suppress_small=True)}\n"

def draw_matches_with_inliers(img1, kp1, img2, kp2, matches, inlier_mask=None, max_draw=100):
    if inlier_mask is not None:
        selected = [m for i, m in enumerate(matches) if inlier_mask[i]]
    else:
        selected = matches

    selected = sorted(selected, key=lambda m: m.distance)[:max_draw]

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, selected, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis

def draw_epilines_horizontal_check(rect_left, rect_right, num_lines=20):
    if len(rect_left.shape) == 2:
        left_vis = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = rect_left.copy()

    if len(rect_right.shape) == 2:
        right_vis = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = rect_right.copy()

    h, w = left_vis.shape[:2]
    canvas = np.hstack([left_vis, right_vis])

    ys = np.linspace(20, h - 20, num_lines).astype(int)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for i, y in enumerate(ys):
        color = colors[i % len(colors)]
        cv2.line(canvas, (0, y), (2 * w - 1, y), color, 1)

    return canvas

# ------------------------------------------------------------
# Feature matching
# ------------------------------------------------------------

def compute_feature_matches(img1_gray, img2_gray, method="sift", max_features=4000, ratio_thresh=0.6):
    method = method.lower()

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError("Unsupported feature method. Use 'sift' or 'orb'.")

    kp1, des1 = detector.detectAndCompute(img1_gray, None)
    kp2, des2 = detector.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Could not detect enough features in one or both images.")

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        raise RuntimeError(f"Not enough good matches after ratio test: {len(good_matches)}")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return kp1, kp2, good_matches, pts1, pts2

# ------------------------------------------------------------
# Fundamental / Essential / Pose
# ------------------------------------------------------------

def estimate_fundamental_matrix(pts1, pts2, ransac_thresh=0.5, confidence=0.999):
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence
    )

    if F is None or F.shape != (3, 3):
        raise RuntimeError("Fundamental matrix estimation failed.")

    return F, mask.ravel().astype(bool)

def build_approx_intrinsic_matrix(width, height, fx_px=None, fy_px=None, cx=None, cy=None):
    if fx_px is None:
        fx_px = 0.9 * width
    if fy_px is None:
        fy_px = 0.9 * width
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    K = np.array([
        [fx_px, 0, cx],
        [0, fy_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K

def compute_essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1

    U, S, Vt = np.linalg.svd(E)
    S_new = np.array([1.0, 1.0, 0.0])
    E_rank2 = U @ np.diag(S_new) @ Vt
    return E_rank2

def recover_pose_from_E(E, K, pts1, pts2, inlier_mask):
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    if len(pts1_in) < 8:
        raise RuntimeError("Not enough inlier points for pose recovery.")

    _, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, pose_mask

# ------------------------------------------------------------
# Rectification / Disparity
# ------------------------------------------------------------

def rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F):
    h, w = img1.shape[:2]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1_in),
        np.float32(pts2_in),
        F,
        imgSize=(w, h)
    )

    if not ok:
        raise RuntimeError("Uncalibrated rectification failed.")

    rect1 = cv2.warpPerspective(img1, H1, (w, h))
    rect2 = cv2.warpPerspective(img2, H2, (w, h))

    return rect1, rect2, H1, H2

def compute_disparity_sgbm(rect1_gray, rect2_gray):
    # light smoothing only
    rect1_gray = cv2.GaussianBlur(rect1_gray, (3, 3), 0)
    rect2_gray = cv2.GaussianBlur(rect2_gray, (3, 3), 0)

    min_disp = 0
    num_disp = 16 * 8   # 128, must be multiple of 16
    block_size = 7      # smaller block for finer detail

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=16,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(rect1_gray, rect2_gray).astype(np.float32) / 16.0
    disparity[disparity <= min_disp] = np.nan
    return disparity

def normalize_disparity_for_display(disparity):
    disp = disparity.copy()

    # remove invalid
    disp[disp <= 0] = np.nan

    # percentile scaling (VERY IMPORTANT)
    min_val = np.nanpercentile(disp, 5)
    max_val = np.nanpercentile(disp, 95)

    disp = (disp - min_val) / (max_val - min_val + 1e-6)
    disp = np.clip(disp, 0, 1)

    disp = (disp * 255).astype(np.uint8)

    return disp

# ------------------------------------------------------------
# Point selection
# ------------------------------------------------------------

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def get_user_selected_point(image_bgr, window_name="Click object center and press Enter"):
    global clicked_point
    clicked_point = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        show = image_bgr.copy()
        if clicked_point is not None:
            cv2.circle(show, clicked_point, 8, (0, 0, 255), -1)
            cv2.putText(
                show, f"Point: {clicked_point}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                show, "Press ENTER to confirm", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
        else:
            cv2.putText(
                show, "Click object center, then press ENTER", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow(window_name, show)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 or key == 10:
            if clicked_point is not None:
                break
        elif key == 27:
            clicked_point = None
            break

    cv2.destroyWindow(window_name)
    return clicked_point

def local_median_disparity(disparity, x, y, window=31):
    h, w = disparity.shape
    r = window // 2

    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)

    patch = disparity[y1:y2, x1:x2]
    valid = patch[patch > 0]

    if len(valid) == 0:
        return None

    return float(np.median(valid))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uncalibrated stereo setup with F, E, R, disparity, and distance estimation.")
    parser.add_argument("--left", required=True, help="Path to left image")
    parser.add_argument("--right", required=True, help="Path to right image")
    parser.add_argument("--output_dir", default="results", help="Output folder")

    parser.add_argument("--baseline_cm", type=float, required=True,
                        help="Camera shift between two captures in cm")

    parser.add_argument("--ground_truth_cm", type=float, default=None,
                        help="Measured real distance from camera to selected object in cm")

    parser.add_argument("--fx_px", type=float, default=None,
                        help="Approximate focal length fx in pixels")
    parser.add_argument("--fy_px", type=float, default=None,
                        help="Approximate focal length fy in pixels")

    parser.add_argument("--feature", choices=["sift", "orb"], default="sift",
                        help="Feature detector to use")
    parser.add_argument("--ratio_thresh", type=float, default=0.6,
                        help="Ratio test threshold")
    parser.add_argument("--ransac_thresh", type=float, default=0.5,
                        help="RANSAC reprojection threshold")

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    img1 = cv2.imread(args.left)
    img2 = cv2.imread(args.right)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read left or right image.")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1_gray.shape[:2]

    print("\n[1] Detecting and matching features...")
    kp1, kp2, matches, pts1, pts2 = compute_feature_matches(
        img1_gray,
        img2_gray,
        method=args.feature,
        ratio_thresh=args.ratio_thresh
    )
    print(f"Total good matches after ratio test: {len(matches)}")

    raw_match_vis = draw_matches_with_inliers(img1, kp1, img2, kp2, matches, None, max_draw=120)
    cv2.imwrite(os.path.join(args.output_dir, "01_raw_matches.png"), raw_match_vis)

    print("\n[2] Estimating Fundamental Matrix...")
    F, inlier_mask = estimate_fundamental_matrix(
        pts1, pts2,
        ransac_thresh=args.ransac_thresh,
        confidence=0.999
    )
    num_inliers = int(np.sum(inlier_mask))
    print(f"Inliers after RANSAC: {num_inliers} / {len(matches)}")

    if num_inliers < 20:
        raise RuntimeError(
            f"Too few inliers ({num_inliers}). Rectification will likely fail."
        )

    inlier_match_vis = draw_matches_with_inliers(img1, kp1, img2, kp2, matches, inlier_mask, max_draw=120)
    cv2.imwrite(os.path.join(args.output_dir, "02_inlier_matches.png"), inlier_match_vis)

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    print("\n[3] Building approximate intrinsic matrix K...")
    K1 = build_approx_intrinsic_matrix(w, h, args.fx_px, args.fy_px)
    K2 = K1.copy()
    print(K1)

    print("\n[4] Computing Essential Matrix E...")
    E = compute_essential_matrix(F, K1, K2)

    print("\n[5] Recovering pose (R, t)...")
    R, t, pose_mask = recover_pose_from_E(E, K1, pts1, pts2, inlier_mask)

    print("\n[6] Rectifying images...")
    rect1_bgr, rect2_bgr, H1, H2 = rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F)
    rect1_gray = cv2.cvtColor(rect1_bgr, cv2.COLOR_BGR2GRAY)
    rect2_gray = cv2.cvtColor(rect2_bgr, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(args.output_dir, "03_rectified_left.png"), rect1_bgr)
    cv2.imwrite(os.path.join(args.output_dir, "04_rectified_right.png"), rect2_bgr)

    rect_check = draw_epilines_horizontal_check(rect1_bgr, rect2_bgr, num_lines=20)
    cv2.imwrite(os.path.join(args.output_dir, "05_rectification_check.png"), rect_check)

    print("\n[7] Computing disparity...")
    disparity = compute_disparity_sgbm(rect1_gray, rect2_gray)
    disp_vis = normalize_disparity_for_display(disparity)
    cv2.imwrite(os.path.join(args.output_dir, "06_disparity.png"), disp_vis)

    print("\n[8] Select the object point on rectified LEFT image...")
    clicked = get_user_selected_point(rect1_bgr, "Click object center and press Enter")

    if clicked is None:
        raise RuntimeError("No point selected. Press ESC cancelled selection.")

    x, y = clicked
    disp_value = local_median_disparity(disparity, x, y, window=31)

    if disp_value is None or disp_value <= 0:
        raise RuntimeError(
            "No valid disparity found around selected point. Click a more textured region."
        )

    fx = K1[0, 0]
    baseline_cm = args.baseline_cm

    estimated_distance_cm = (fx * baseline_cm) / disp_value

    print("\n[9] Distance estimation complete.")
    print(f"Selected point: ({x}, {y})")
    print(f"Median disparity: {disp_value:.3f} px")
    print(f"Estimated distance: {estimated_distance_cm:.3f} cm")

    annotated = rect1_bgr.copy()
    cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)

    info_lines = [
        f"Selected point: ({x}, {y})",
        f"Disparity: {disp_value:.2f} px",
        f"Estimated distance: {estimated_distance_cm:.2f} cm"
    ]

    abs_error = None
    rel_error = None
    if args.ground_truth_cm is not None:
        abs_error = abs(estimated_distance_cm - args.ground_truth_cm)
        rel_error = (abs_error / args.ground_truth_cm) * 100 if args.ground_truth_cm > 0 else None
        info_lines.append(f"Ground truth: {args.ground_truth_cm:.2f} cm")
        info_lines.append(f"Absolute error: {abs_error:.2f} cm")
        if rel_error is not None:
            info_lines.append(f"Relative error: {rel_error:.2f}%")

    for i, line in enumerate(info_lines):
        cv2.putText(
            annotated, line, (20, 35 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

    cv2.imwrite(os.path.join(args.output_dir, "07_annotated_result.png"), annotated)

    report_text = []
    report_text.append("Uncalibrated Stereo Distance Estimation Results\n")
    report_text.append("=" * 60 + "\n")
    report_text.append(format_matrix("K", K1))
    report_text.append(format_matrix("F", F))
    report_text.append(format_matrix("E", E))
    report_text.append(format_matrix("R", R))
    report_text.append(format_matrix("t (up to scale)", t))
    report_text.append(f"Image size: {w} x {h}\n")
    report_text.append(f"Feature method: {args.feature}\n")
    report_text.append(f"Ratio threshold: {args.ratio_thresh}\n")
    report_text.append(f"RANSAC threshold: {args.ransac_thresh}\n")
    report_text.append(f"Total good matches: {len(matches)}\n")
    report_text.append(f"RANSAC inliers: {num_inliers}\n")
    report_text.append(f"Baseline (cm): {baseline_cm:.3f}\n")
    report_text.append(f"Approx fx (px): {fx:.3f}\n")
    report_text.append(f"Selected point: ({x}, {y})\n")
    report_text.append(f"Median disparity (px): {disp_value:.3f}\n")
    report_text.append(f"Estimated distance (cm): {estimated_distance_cm:.3f}\n")

    if args.ground_truth_cm is not None:
        report_text.append(f"Ground truth distance (cm): {args.ground_truth_cm:.3f}\n")
        report_text.append(f"Absolute error (cm): {abs_error:.3f}\n")
        if rel_error is not None:
            report_text.append(f"Relative error (%): {rel_error:.3f}\n")

    save_text(os.path.join(args.output_dir, "matrices_and_results.txt"), "".join(report_text))

    md = []
    md.append("# Stereo Report Notes\n")
    md.append("## Summary")
    md.append(textwrap.dedent(f"""
    Two classroom images were captured using the same camera from slightly different horizontal positions,
    forming a pseudo-stereo uncalibrated setup. Feature correspondences were extracted using {args.feature.upper()}.
    The Fundamental Matrix (F) was estimated using RANSAC. Since the camera intrinsics were unknown, an approximate
    intrinsic matrix (K) was assumed from the image resolution to compute the Essential Matrix (E). The Rotation Matrix (R)
    was recovered from E. The image pair was rectified, disparity was computed using StereoSGBM, and the selected object's
    distance was estimated using:

        Z = (f * B) / d

    where Z is distance, f is focal length in pixels, B is baseline in cm, and d is disparity in pixels.
    """).strip())

    md.append("\n## Output Files")
    md.append("- 01_raw_matches.png")
    md.append("- 02_inlier_matches.png")
    md.append("- 03_rectified_left.png")
    md.append("- 04_rectified_right.png")
    md.append("- 05_rectification_check.png")
    md.append("- 06_disparity.png")
    md.append("- 07_annotated_result.png")
    md.append("- matrices_and_results.txt")

    save_text(os.path.join(args.output_dir, "report_notes.md"), "\n".join(md))

    print(f"\nDone. Outputs saved in: {args.output_dir}")

if __name__ == "__main__":
    main()