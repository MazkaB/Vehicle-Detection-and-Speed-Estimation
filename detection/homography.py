import cv2
import numpy as np
import math

def compute_homography_and_scale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blur, lower, upper)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

    if linesP is None:
        H = np.eye(3, dtype=np.float32)
        scale = 1.0
        return H, scale

    frame_h, frame_w = frame.shape[:2]
    left_lines = []
    right_lines = []

    for line in linesP:
        for x1, y1, x2, y2 in line:
            if (x2 - x1) == 0:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if slope < -0.3:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:
                right_lines.append((x1, y1, x2, y2))

    left_xs = []
    for (x1, y1, x2, y2) in left_lines:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        x_bottom = (frame_h - b) / m
        left_xs.append(x_bottom)

    right_xs = []
    for (x1, y1, x2, y2) in right_lines:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        x_bottom = (frame_h - b) / m
        right_xs.append(x_bottom)

    if len(left_xs) == 0:
        left_boundary = frame_w * 0.25
    else:
        left_boundary = np.median(left_xs)

    if len(right_xs) == 0:
        right_boundary = frame_w * 0.75
    else:
        right_boundary = np.median(right_xs)

    def average_line(lines):
        if len(lines) == 0:
            return None
        x1s, y1s, x2s, y2s = zip(*lines)
        return (np.mean(x1s), np.mean(y1s), np.mean(x2s), np.mean(y2s))

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (px, py)

    if left_avg is not None and right_avg is not None:
        vp = line_intersection(left_avg, right_avg)
        if vp is None:
            vp = (frame_w / 2, frame_h / 2)
    else:
        vp = (frame_w / 2, frame_h / 2)

    src_pts = np.array([
        [left_boundary, frame_h],
        [right_boundary, frame_h],
        [right_boundary, vp[1]],
        [left_boundary, vp[1]]
    ], dtype=np.float32)

    REAL_LANE_WIDTH = 20 # definisi lebar jalan
    dst_width = right_boundary - left_boundary
    dst_height = frame_h - vp[1]
    dst_pts = np.array([
        [0, dst_height],
        [dst_width, dst_height],
        [dst_width, 0],
        [0, 0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    scale = REAL_LANE_WIDTH / dst_width

    return H, scale


def transform_point(pt, H, scale):
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float32)
    dst = H.dot(vec)
    dst /= dst[2]
    world_x = dst[0] * scale
    world_y = dst[1] * scale
    return world_x, world_y
