import sys as _sys
if _sys.argv[0].endswith("app.py") and "streamlit.runtime.scriptrunner" not in _sys.modules:
    import subprocess as _sp
    _sp.run([_sys.executable, "-m", "streamlit", "run", __file__] + _sys.argv[1:])
    _sys.exit(0)

import io
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tái tạo Chữ ký B-Spline",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Sidebar header ── */
.sidebar-header {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    border-radius: 12px;
    padding: 18px 16px;
    margin-bottom: 20px;
    text-align: center;
}
.sidebar-header h1 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff !important;
    margin: 0;
    letter-spacing: 0.5px;
}
.sidebar-header p {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.75) !important;
    margin: 4px 0 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #388bfd !important;
    margin: 20px 0 8px;
}

/* ── Dashboard title ── */
.dashboard-title {
    background: linear-gradient(90deg, #1f6feb, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.dashboard-subtitle {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 4px;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}
.glass-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #c9d1d9;
    margin: 0 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metric pill ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.metric-pill {
    background: linear-gradient(135deg, #1f2937, #111827);
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 10px 16px;
    min-width: 110px;
    text-align: center;
}
.metric-pill .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-pill .label {
    font-size: 0.7rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ── Pipeline step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #1f6feb22, #388bfd11);
    border: 1px solid #1f6feb55;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #79c0ff;
    margin-bottom: 10px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #1f6feb66 !important;
    border-radius: 16px !important;
    background: rgba(31, 111, 235, 0.05) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #388bfd !important;
    background: rgba(31, 111, 235, 0.1) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(31,111,235,0.3) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(31,111,235,0.5) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
}

/* ── Success / info banners ── */
.stAlert { border-radius: 10px !important; }

/* ── Tabs ── */
[data-testid="stTab"] {
    font-weight: 500 !important;
    color: #8b949e !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #1f6feb !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1f6feb, #58a6ff) !important;
}

/* ── Log box ── */
.log-box {
    background: #010409;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.78rem;
    color: #7ee787;
    max-height: 260px;
    overflow-y: auto;
    line-height: 1.7;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Matplotlib figures ── */
.stImage img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — IMAGE PROCESSING  (image_processing.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def choose_next_point(current, neighbors, ordered_points):
    """Direction-aware greedy next-pixel selector."""
    if len(ordered_points) < 2:
        return neighbors[0]
    prev = ordered_points[-2]
    dir_x = current[0] - prev[0]
    dir_y = current[1] - prev[1]
    best_point, best_score = neighbors[0], -999999
    for cand in neighbors:
        step_x = cand[0] - current[0]
        step_y = cand[1] - current[1]
        score = dir_x * step_x + dir_y * step_y
        if score > best_score:
            best_score = score
            best_point = cand
    return best_point


def extract_smooth_skeleton(skel):
    """
    Branch-tracing DFS that extracts ordered pixels from a skeletonised binary
    image without any overlap (anti-overlapping guarantee).
    """
    y_idx, x_idx = np.where(skel == 255)
    points = set(zip(x_idx.tolist(), y_idx.tolist()))
    ordered_points = []

    while points:
        # Prefer end-points (exactly 1 neighbour) as starting seeds
        start_p = next(iter(points))
        for p in points:
            neighbours = sum(
                1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                if (dx != 0 or dy != 0) and (p[0] + dx, p[1] + dy) in points
            )
            if neighbours == 1:
                start_p = p
                break

        current = start_p
        points.remove(current)
        ordered_points.append(current)

        while True:
            neighbours = [
                (current[0] + dx, current[1] + dy)
                for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                if (dx != 0 or dy != 0) and (current[0] + dx, current[1] + dy) in points
            ]
            if not neighbours:
                break
            current = choose_next_point(current, neighbours, ordered_points)
            points.remove(current)
            ordered_points.append(current)

    return ordered_points


def process_image_pipeline(img_bytes: bytes, log: list) -> tuple:
    """
    Full image-processing pipeline.

    Returns
    -------
    original_img  : np.ndarray  (grayscale)
    skel          : np.ndarray  (binary skeleton)
    ordered_pts   : list[(x, y)]
    diempixel_dat : str         (content of diempixel.dat)
    """
    log.append("📷  Đang giải mã ảnh đã tải lên …")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể giải mã ảnh. Vui lòng tải lên file JPG/PNG hợp lệ.")

    log.append(f"✅  Ảnh đã tải — kích thước: {img.shape[1]}×{img.shape[0]} px")

    # ── Gaussian blur + Otsu thresholding ──────────────────────────────────
    log.append("🔧  Áp dụng Gaussian blur + ngưỡng hóa Otsu …")
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Morphological close + dilate ───────────────────────────────────────
    log.append("🔧  Xử lý hình thái học (đóng + giãn nở) …")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    thick_binary = cv2.dilate(closed, kernel, iterations=1)

    # ── Zhang-Suen thinning ────────────────────────────────────────────────
    log.append("🦴  Đang làm mỏng theo thuật toán Zhang-Suen …")
    skel = cv2.ximgproc.thinning(thick_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skel_px = int(np.sum(skel == 255))
    log.append(f"✅  Bộ khung xương đã trích xuất — {skel_px:,} pixel khung xương")

    # ── Branch-tracing DFS ─────────────────────────────────────────────────
    log.append("🔍  Theo dõi nhánh khung xương (DFS) …")
    ordered_pts = extract_smooth_skeleton(skel)
    log.append(f"✅  Trích xuất {len(ordered_pts):,} điểm pixel có thứ tự")

    # ── Build diempixel.dat content ────────────────────────────────────────
    log.append("💾  Định dạng diempixel.dat …")
    h = img.shape[0]
    lines = [f"{len(ordered_pts)}\n"]
    for x, y in ordered_pts:
        lines.append(f"{float(x):.2f} {float(h - y):.2f} 0.00 1.00\n")
    diempixel_dat = "".join(lines)
    log.append("✅  diempixel.dat đã sẵn sàng")

    return img, skel, ordered_pts, diempixel_dat


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — B-SPLINE MATHEMATICS  (B_Spline_LST_Math.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def read_data(diempixel_content: str) -> np.ndarray:
    """
    Required function 1 (academic).
    Reads pixel-point data from the diempixel.dat string content.

    Returns
    -------
    Q : np.ndarray, shape (m, 3)   — [X, Y, Z] columns
    """
    lines = diempixel_content.strip().splitlines()
    Q = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            Q.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not Q:
        raise ValueError("diempixel.dat contains no data points.")
    return np.array(Q)


def chord_length_parameterization(Q: np.ndarray) -> np.ndarray:
    """Chord-length parameterisation → ubar ∈ [0, 1]."""
    m = len(Q)
    distances = np.zeros(m)
    total_length = 0.0
    for i in range(1, m):
        distances[i] = np.linalg.norm(Q[i] - Q[i - 1])
        total_length += distances[i]
    if total_length == 0:
        raise ValueError("Total chord length is 0 — degenerate stroke.")
    ubar = np.zeros(m)
    ubar[-1] = 1.0
    for i in range(1, m - 1):
        ubar[i] = ubar[i - 1] + distances[i] / total_length
    return ubar


def generate_knot_vector(n: int, p: int) -> np.ndarray:
    """Generate a clamped uniform knot vector of size n+p+1."""
    m_knots = n + p + 1
    U = np.zeros(m_knots)
    for i in range(p + 1):
        U[i] = 0.0
        U[m_knots - 1 - i] = 1.0
    for i in range(p + 1, n):
        U[i] = (i - p) / (n - p)
    return U


def Nip(i: int, p: int, u: float, U: np.ndarray) -> float:
    """
    Required function 2 (academic).
    Cox-de Boor recursion for the i-th B-spline basis function of degree p
    evaluated at parameter u, given knot vector U.
    """
    if p == 0:
        if U[i] <= u < U[i + 1]:
            return 1.0
        if u == 1.0 and U[i + 1] == 1.0:
            return 1.0
        return 0.0

    left = 0.0
    denom_left = U[i + p] - U[i]
    if denom_left != 0:
        left = ((u - U[i]) / denom_left) * Nip(i, p - 1, u, U)

    right = 0.0
    denom_right = U[i + p + 1] - U[i + 1]
    if denom_right != 0:
        right = ((U[i + p + 1] - u) / denom_right) * Nip(i + 1, p - 1, u, U)

    return left + right


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — RECONSTRUCTION  (bspline_reconstruct_khang.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def read_and_split_strokes(diempixel_content: str, jump_threshold: float = 2.0) -> list:
    """
    Reads all pixel points from the diempixel.dat content string,
    splits them into individual strokes by distance threshold,
    applies Spur Pruning and Junction Healing.

    Returns
    -------
    strokes : list of np.ndarray, each shape (m_i, 3)
    """
    lines = diempixel_content.strip().splitlines()
    Q_all = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            Q_all.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))

    if not Q_all:
        return []

    strokes = []
    current_stroke = [Q_all[0]]

    for i in range(1, len(Q_all)):
        dist = np.linalg.norm(Q_all[i] - Q_all[i - 1])
        if dist > jump_threshold:
            # Spur Pruning: discard strokes shorter than 15 points
            if len(current_stroke) > 15:
                strokes.append(np.array(current_stroke))
            current_stroke = [Q_all[i]]
        else:
            current_stroke.append(Q_all[i])

    if len(current_stroke) > 15:
        strokes.append(np.array(current_stroke))

    # Junction Healing: snap near-endpoints to closest point on other strokes
    for i in range(len(strokes)):
        for end_idx in [0, -1]:
            best_dist = float("inf")
            best_target = None
            for j in range(len(strokes)):
                if i == j:
                    continue
                dists = np.linalg.norm(strokes[j] - strokes[i][end_idx], axis=1)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < best_dist:
                    best_dist = dists[min_idx]
                    best_target = strokes[j][min_idx]
            if best_dist is not None and best_dist <= 5.0:
                strokes[i][end_idx] = best_target

    return strokes


def LSTBSplineReconstruction(Q: np.ndarray, degree: int = 3,
                              smooth_lambda: float = 0.001) -> tuple:
    """
    Required function 3 (academic).
    Regularised Least-Squares B-Spline Reconstruction.

    Academic variables:
        Unum    = n                (int)   — number of control points
        Udegree = degree           (int)   — curve degree
        Uknot   = U                (array) — knot vector  (double *Uknot)
        P4      = P with W=1.0    (array) — homogeneous control points (double *P4)

    Parameters
    ----------
    Q             : data points, shape (m, 3)
    degree        : B-spline degree  (Udegree)
    smooth_lambda : regularisation weight λ for smoothing

    Returns
    -------
    P : np.ndarray, shape (n, 3) — control points  (Unum × 3 coords)
    U : np.ndarray, shape (n+degree+1,) — knot vector (Uknot)
    """
    m = len(Q)

    # Dynamic control-point count: longer strokes get more control points
    Unum = min(80, max(degree + 1, m // 3 + 2))   # int Unum
    n = Unum                                        # alias for clarity
    Udegree = degree                                # int Udegree

    ubar = chord_length_parameterization(Q)
    Uknot = generate_knot_vector(n, Udegree)       # double *Uknot

    # Build basis matrix N  (m × n)
    N = np.zeros((m, n))
    for k in range(m):
        for i_ctrl in range(n):
            N[k, i_ctrl] = Nip(i_ctrl, Udegree, ubar[k], Uknot)

    # ── Second-order difference matrix D2 (n-2 × n) ──────────────────────
    # Penalises curvature of the control polygon (Δ²Pᵢ = Pᵢ - 2Pᵢ₊₁ + Pᵢ₊₂)
    # Unlike λ·I which shrinks control points toward 0 (causing curve collapse),
    # D2ᵀD2 only penalises bending → curves stay large but become smoother.
    D2 = np.zeros((max(1, n - 2), n))
    for i in range(n - 2):
        D2[i, i]     =  1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] =  1.0

    # Scale λ by data variance so the slider is scale-independent
    data_scale = float(np.mean(np.var(Q[:, :2], axis=0))) + 1e-8

    # Regularised normal equations:  (NᵀN + λ·scale·D2ᵀD2) P = NᵀQ
    NT_N = np.dot(N.T, N)
    NT_Q = np.dot(N.T, Q)
    A = NT_N + smooth_lambda * data_scale * np.dot(D2.T, D2)

    # Solve for P (double *P4 — X, Y, Z; W is always 1.0)
    P = np.linalg.lstsq(A, NT_Q, rcond=None)[0]

    return P, Uknot  # Unum = len(P), Udegree preserved in caller


def export_dutmod_multiple_curves(curves: list, degree: int) -> str:
    """
    Required function 4 (academic).
    Formats the DUTMod/DISCO bsplinecurve.dat file content for multiple curves.

    Academic variables in output:
        Unum      — number of control points per curve
        Udegree   — B-spline degree
        UKnotType — always 1 (clamped uniform)
        P4        — control point X Y Z W (W=1.00000000)
        Uknot     — knot values

    Precision rules:
        - X, Y, Z  → exactly 2 decimal places  (%.2f)
        - W        → always 1.00000000 (8 decimal places)
        - Uknot    → 8 decimal places (%.8f)
        - NO intermediate rounding; only final string formatting applies rounding
    """
    lines = []
    for idx, (P, U) in enumerate(curves):
        Unum    = len(P)    # int Unum
        Udegree = degree    # int Udegree

        lines.append("==========================\n")
        lines.append("\n[BSPLINECURVE]\n\n")
        lines.append(f"{Unum}, {Udegree}, 1 // UNum, UDegree, UKnotType (Curve {idx + 1})\n\n")
        lines.append("// Control Points\n")

        for ctrl_pt in P:
            # double *P4 : X, Y, Z (%.2f) and W=1.00000000
            x, y, z = ctrl_pt          # no intermediate rounding here
            lines.append(f"{x:.2f} {y:.2f} {z:.2f} 1.00000000 0\n")

        lines.append("\n// UKnot\n")
        for u_val in U:
            lines.append(f"{u_val:.8f}\n")    # double *Uknot — 8 decimal places
        lines.append("\n")

    return "".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def plot_original(img: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
    ax.imshow(img, cmap="gray")
    ax.set_title("Ảnh gốc", color="#c9d1d9", fontsize=11, fontweight="600", pad=10)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


def plot_skeleton(skel: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
    display = np.zeros((*skel.shape, 3), dtype=np.uint8)
    display[skel == 255] = [88, 166, 255]
    ax.imshow(display)
    ax.set_title("Khung xương (Zhang-Suen)", color="#c9d1d9",
                 fontsize=11, fontweight="600", pad=10)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


def plot_extracted_points(ordered_pts: list, img_shape: tuple) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
    ax.set_facecolor("#010409")
    if ordered_pts:
        xs = [p[0] for p in ordered_pts]
        ys = [p[1] for p in ordered_pts]
        sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="plasma",
                        s=1.5, alpha=0.8, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, label="Thứ tự", shrink=0.8)
        cb.ax.yaxis.label.set_color("#8b949e")
        cb.ax.tick_params(colors="#6e7681")
    ax.set_xlim(0, img_shape[1])
    ax.set_ylim(img_shape[0], 0)
    ax.set_title("Điểm được trích xuất & sắp xếp", color="#c9d1d9",
                 fontsize=11, fontweight="600", pad=10)
    ax.tick_params(colors="#6e7681")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


def plot_bspline_curves(strokes: list, curves: list, degree: int,
                        img_shape: tuple) -> bytes:
    """Render đường cong B-spline tái tạo trên nền trắng giống chữ ký gốc."""
    # Màu mực chữ ký (xanh đậm như mực bút bi)
    INK_COLORS = [
        "#3a3ab0", "#1a1a8c", "#2525a0", "#4040c0",
        "#2828b0", "#1515a0", "#3535b5", "#2020a8",
    ]

    h_img, w_img = img_shape
    fig, ax = plt.subplots(figsize=(w_img / 100, h_img / 100),
                           facecolor="white", dpi=100)
    ax.set_facecolor("white")

    for idx, ((P, U), stroke) in enumerate(zip(curves, strokes)):
        col = INK_COLORS[idx % len(INK_COLORS)]
        n = len(P)

        # Dữ liệu điểm điều khiển đã ở hệ toạ độ toán học (y=0 ở dưới),
        # khớp với trục Y mặc định của matplotlib (ylim(0,h)) → không cần đảo.
        t_vals = np.linspace(0, 1, max(500, len(stroke) * 3))
        curve_pts = []
        for t in t_vals:
            pt = np.zeros(3)
            for i_ctrl in range(n):
                pt += Nip(i_ctrl, degree, t, U) * P[i_ctrl]
            curve_pts.append(pt)
        curve_pts = np.array(curve_pts)

        # Vẽ đường cong mịn, không có đa giác điều khiển
        ax.plot(curve_pts[:, 0], curve_pts[:, 1],
                color=col, linewidth=1.5, alpha=0.92,
                solid_capstyle="round", solid_joinstyle="round")

    ax.set_xlim(0, w_img)
    ax.set_ylim(0, h_img)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


def plot_bspline_curves_dark(strokes: list, curves: list, degree: int,
                             img_shape: tuple) -> bytes:
    """Render đường cong B-spline tái tạo trên nền tối (chế độ kỹ thuật)."""
    COLORS = [
        "#58a6ff", "#f78166", "#56d364", "#e3b341",
        "#bc8cff", "#ff7b72", "#79c0ff", "#ffa657",
    ]
    h_img, w_img = img_shape
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#010409")

    for idx, ((P, U), stroke) in enumerate(zip(curves, strokes)):
        col = COLORS[idx % len(COLORS)]
        n = len(P)

        # Dữ liệu đã ở hệ toạ độ toán học → vẽ trực tiếp, không cần đảo Y
        ax.scatter(stroke[:, 0], stroke[:, 1], s=1, color=col,
                   alpha=0.25, linewidths=0)

        t_vals = np.linspace(0, 1, max(500, len(stroke) * 3))
        curve_pts = []
        for t in t_vals:
            pt = np.zeros(3)
            for i_ctrl in range(n):
                pt += Nip(i_ctrl, degree, t, U) * P[i_ctrl]
            curve_pts.append(pt)
        curve_pts = np.array(curve_pts)

        ax.plot(curve_pts[:, 0], curve_pts[:, 1],
                color=col, linewidth=1.6, alpha=0.95,
                label=f"Đường {idx + 1}  (n={n})")

        ax.plot(P[:, 0], P[:, 1], "--", color=col, linewidth=0.7, alpha=0.35)
        ax.scatter(P[:, 0], P[:, 1], s=14, color=col,
                   alpha=0.6, zorder=5, linewidths=0)

    ax.set_xlim(0, w_img)
    ax.set_ylim(0, h_img)
    ax.set_title("Đường cong B-Spline tái tạo (chế độ kỹ thuật)", color="#c9d1d9",
                 fontsize=12, fontweight="700", pad=12)
    ax.tick_params(colors="#6e7681", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    legend = ax.legend(fontsize=7.5, framealpha=0.15, labelcolor="#c9d1d9",
                       edgecolor="#21262d", loc="best")
    legend.get_frame().set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()
    data = fig_to_bytes(fig)
    plt.close(fig)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h1>✍️ Tái tạo B-Spline</h1>
        <p>ĐUT · DHMT · Đồ họa máy tính</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">📂 Đầu vào</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Tải lên ảnh chữ ký",
        type=["jpg", "jpeg", "png"],
        help="Kéo thả hoặc nhấn để tải lên ảnh chữ ký JPG/PNG.",
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">⚙️ Tham số</p>', unsafe_allow_html=True)

    smooth_lambda = st.slider(
        "Hệ số làm mịn Lambda (λ)",
        min_value=0.000, max_value=1.000,
        value=0.001, step=0.001,
        format="%.3f",
        help="Trọng số chính quy hóa trong (N^T N + λ I) P = N^T Q. "
             "λ càng lớn → đường cong càng mịn nhưng kém chính xác hơn.",
    )

    jump_threshold = st.slider(
        "Ngưỡng nhảy (px)",
        min_value=1.0, max_value=10.0,
        value=2.0, step=0.5,
        format="%.1f",
        help="Khoảng cách pixel mà tại đó bộ theo dõi bắt đầu một nét mới.",
    )

    st.markdown('<p class="section-label">🎓 Thông tin học thuật</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#6e7681; line-height:1.7;">
        <b style="color:#58a6ff;">Unum</b> — số điểm điều khiển<br>
        <b style="color:#58a6ff;">Udegree</b> — bậc đường cong (3)<br>
        <b style="color:#58a6ff;">Uknot</b> — vector nút kẹp<br>
        <b style="color:#58a6ff;">P4</b> — điểm điều khiển X Y Z W<br>
        <b style="color:#58a6ff;">λ</b> — hệ số làm mịn chính quy<br>
        <b style="color:#58a6ff;">Nip()</b> — đệ quy Cox-de Boor<br>
        <b style="color:#58a6ff;">LST</b> — Bình phương nhỏ nhất chính quy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("🚀  Chạy toàn bộ Pipeline", use_container_width=True,
                        type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 8px 0 24px;">
    <p class="dashboard-title">Tái tạo Chữ ký B-Spline</p>
    <p class="dashboard-subtitle">
        Trích xuất đa nét · LST chính quy hóa · Xuất DUTMod/DISCO
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pipeline, tab_result, tab_log = st.tabs(
    ["🖼️  Pipeline trực quan", "📈  Kết quả B-Spline", "📋  Nhật ký xử lý"]
)

# ── Session state placeholders ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Run pipeline when button pressed ──────────────────────────────────────
if run_btn:
    if uploaded_file is None:
        st.warning("⚠️  Vui lòng tải lên ảnh chữ ký trước.")
    else:
        log_lines = []
        progress = st.progress(0, text="Đang khởi tạo …")

        try:
            # Bước 1 — Xử lý ảnh
            progress.progress(10, text="Bước 1/3  — Xử lý ảnh …")
            img_bytes = uploaded_file.read()
            original_img, skel, ordered_pts, diempixel_content = process_image_pipeline(
                img_bytes, log_lines
            )
            progress.progress(40, text="Bước 2/3  — Tách nét & LST …")

            # Bước 2 — Tách nét
            log_lines.append(f"✂️  Tách nét (ngưỡng nhảy={jump_threshold}) …")
            strokes = read_and_split_strokes(diempixel_content, jump_threshold)
            log_lines.append(f"✅  Phát hiện {len(strokes)} nét sau khi lọc spur")

            # Bước 3 — LSTBSplineReconstruction cho từng nét
            degree = 3
            curves = []
            for s_idx, stroke_pts in enumerate(strokes):
                log_lines.append(
                    f"🔢  Nét {s_idx + 1}: {len(stroke_pts)} điểm → "
                    f"LSTBSplineReconstruction(bậc={degree}, λ={smooth_lambda}) …"
                )
                P, U = LSTBSplineReconstruction(stroke_pts, degree, smooth_lambda)
                curves.append((P, U))
                log_lines.append(
                    f"    Unum={len(P)}, |Uknot|={len(U)}"
                )

            progress.progress(80, text="Bước 3/3  — Định dạng file đầu ra …")

            # Bước 4 — Định dạng bsplinecurve.dat
            log_lines.append("📝  Định dạng bsplinecurve.dat (DUTMod/DISCO) …")
            bsplinecurve_content = export_dutmod_multiple_curves(curves, degree)
            log_lines.append("✅  bsplinecurve.dat đã sẵn sàng")

            progress.progress(100, text="Hoàn tất! ✅")

            # Lưu kết quả vào session state
            st.session_state.results = {
                "original_img": original_img,
                "skel": skel,
                "ordered_pts": ordered_pts,
                "diempixel_content": diempixel_content,
                "strokes": strokes,
                "curves": curves,
                "bsplinecurve_content": bsplinecurve_content,
                "degree": degree,
                "log_lines": log_lines,
            }
            st.success(f"✅  Pipeline hoàn tất — tái tạo {len(strokes)} đường cong!")

        except Exception as exc:
            progress.empty()
            st.error(f"❌  Lỗi pipeline: {exc}")
            log_lines.append(f"❌  LỖI: {exc}")
            st.session_state.results = {"log_lines": log_lines, "error": True}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISUAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    if st.session_state.results and not st.session_state.results.get("error"):
        r = st.session_state.results

        # Hàng số liệu
        n_pts    = len(r["ordered_pts"])
        n_strks  = len(r["strokes"])
        n_ctrl   = sum(len(P) for P, _ in r["curves"])
        h, w     = r["original_img"].shape

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <div class="value">{n_pts:,}</div>
                <div class="label">Số pixel</div>
            </div>
            <div class="metric-pill">
                <div class="value">{n_strks}</div>
                <div class="label">Số nét</div>
            </div>
            <div class="metric-pill">
                <div class="value">{n_ctrl}</div>
                <div class="label">Điểm điều khiển</div>
            </div>
            <div class="metric-pill">
                <div class="value">{w}×{h}</div>
                <div class="label">Độ phân giải</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="step-badge">① Ảnh gốc</div>', unsafe_allow_html=True)
            st.image(plot_original(r["original_img"]), use_container_width=True)

        with col2:
            st.markdown('<div class="step-badge">② Khung xương</div>', unsafe_allow_html=True)
            st.image(plot_skeleton(r["skel"]), use_container_width=True)

        with col3:
            st.markdown('<div class="step-badge">③ Điểm có thứ tự</div>',
                        unsafe_allow_html=True)
            st.image(plot_extracted_points(r["ordered_pts"], r["original_img"].shape),
                     use_container_width=True)

        # Tải xuống diempixel.dat
        st.markdown("---")
        st.markdown('<div class="step-badge">💾 Xuất dữ liệu trung gian</div>',
                    unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns([2, 5])
        with col_dl1:
            st.download_button(
                label="⬇️  Tải diempixel.dat",
                data=r["diempixel_content"].encode("utf-8"),
                file_name="diempixel.dat",
                mime="text/plain",
            )
    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color:#6e7681;">
            <div style="font-size:3rem;">🖼️</div>
            <p style="font-size:1rem; margin-top:12px;">
                Tải lên ảnh chữ ký và nhấn
                <strong style="color:#58a6ff;">Chạy toàn bộ Pipeline</strong>
                để xem kết quả trực quan ở đây.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — B-SPLINE RESULT
# ══════════════════════════════════════════════════════════════════════════════
with tab_result:
    if st.session_state.results and not st.session_state.results.get("error"):
        r = st.session_state.results

        st.markdown('<div class="step-badge">④ Tái tạo B-Spline</div>',
                    unsafe_allow_html=True)

        # Hiển thị hai chế độ xem song song
        col_view1, col_view2 = st.columns(2)
        with col_view1:
            st.markdown("**🖋️ Chữ ký tái tạo (nền trắng)**")
            curve_img = plot_bspline_curves(
                r["strokes"], r["curves"], r["degree"], r["original_img"].shape
            )
            st.image(curve_img, use_container_width=True)
        with col_view2:
            st.markdown("**🔬 Chế độ kỹ thuật (nền tối)**")
            curve_img_dark = plot_bspline_curves_dark(
                r["strokes"], r["curves"], r["degree"], r["original_img"].shape
            )
            st.image(curve_img_dark, use_container_width=True)

        # Chi tiết học thuật từng đường cong
        st.markdown("---")
        st.markdown("#### 📐 Tóm tắt biến học thuật (theo từng đường cong)")
        for idx, (P, U) in enumerate(r["curves"]):
            Unum    = len(P)
            Udegree = r["degree"]
            with st.expander(f"Đường cong {idx + 1}  —  Unum={Unum}, Udegree={Udegree}, |Uknot|={len(U)}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    | Biến | Giá trị |
                    |---|---|
                    | **Unum** (số điểm điều khiển) | `{Unum}` |
                    | **Udegree** (bậc) | `{Udegree}` |
                    | **\\|Uknot\\|** (kích thước vector nút) | `{len(U)}` |
                    | **Smooth λ** | `{smooth_lambda:.3f}` |
                    """)
                with col_b:
                    knot_preview = "  ".join(f"{u:.4f}" for u in U[:8])
                    if len(U) > 8:
                        knot_preview += " …"
                    st.markdown(f"**Uknot (8 giá trị đầu):** `{knot_preview}`")
                    ctrl_preview = "\n".join(
                        f"  {p[0]:.2f}  {p[1]:.2f}  {p[2]:.2f}  1.00000000  0"
                        for p in P[:5]
                    )
                    if len(P) > 5:
                        ctrl_preview += f"\n  … ({len(P) - 5} điểm nữa)"
                    st.code(ctrl_preview, language="text")

        # ── Tải xuống cuối cùng ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="step-badge">⑤ Xuất cuối — DUTMod/DISCO</div>',
                    unsafe_allow_html=True)

        col_p, col_q = st.columns([3, 4])
        with col_p:
            st.download_button(
                label="🏆  Tải bsplinecurve.dat",
                data=r["bsplinecurve_content"].encode("utf-8"),
                file_name="bsplinecurve.dat",
                mime="text/plain",
            )
        with col_q:
            st.caption(
                "Chứa tất cả đường cong B-spline theo định dạng DUTMod/DISCO: "
                "Unum, Udegree, UKnotType, Điểm điều khiển (X Y Z W), giá trị Uknot."
            )

        # Xem trước 50 dòng đầu của file
        with st.expander("📄 Xem trước bsplinecurve.dat (50 dòng đầu)"):
            preview_lines = r["bsplinecurve_content"].splitlines()[:50]
            st.code("\n".join(preview_lines), language="text")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color:#6e7681;">
            <div style="font-size:3rem;">📈</div>
            <p style="font-size:1rem; margin-top:12px;">
                Kết quả tái tạo B-Spline sẽ xuất hiện ở đây sau khi pipeline chạy xong.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROCESSING LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    if st.session_state.results:
        r = st.session_state.results
        log_html = "<br>".join(r.get("log_lines", ["Chưa có nhật ký nào."]))
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color:#6e7681;">
            <div style="font-size:3rem;">📋</div>
            <p style="font-size:1rem; margin-top:12px;">
                Nhật ký xử lý sẽ xuất hiện ở đây sau khi pipeline chạy xong.
            </p>
        </div>
        """, unsafe_allow_html=True)
