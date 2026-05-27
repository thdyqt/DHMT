import numpy as np
import math
# Khang gọi các hàm toán học cơ bản từ file của Hiếu
from B_Spline_LST_Math import chord_length_parameterization, generate_knot_vector, Nip

def read_and_split_strokes(filepath, jump_threshold=2.0):
    Q_all = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                Q_all.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
    
    strokes = []
    current_stroke = [Q_all[0]]
    
    for i in range(1, len(Q_all)):
        dist = np.linalg.norm(Q_all[i] - Q_all[i-1])
        if dist > jump_threshold:
            # 1. LỌC NHIỄU (SPUR PRUNING): 
            # Bỏ qua các nét quá ngắn (<= 15 pixel). Sẽ quét sạch các nét dăm lắt nhắt.
            if len(current_stroke) > 15: 
                strokes.append(np.array(current_stroke))
            current_stroke = [Q_all[i]]
        else:
            current_stroke.append(Q_all[i])
            
    if len(current_stroke) > 15:
        strokes.append(np.array(current_stroke))

    # 2. KHÓA NÚT GIAO (JUNCTION HEALING):
    # Dùng "nam châm" hút các đầu mút bị hở dính chặt lại với nhau
    for i in range(len(strokes)):
        for end_idx in [0, -1]: # Chỉ xét điểm đầu [0] và điểm cuối [-1] của nét
            best_dist = float('inf')
            best_target = None
            
            # Quét tìm điểm gần nhất trên tất cả các nét khác
            for j in range(len(strokes)):
                if i == j: continue
                dists = np.linalg.norm(strokes[j] - strokes[i][end_idx], axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_dist:
                    best_dist = dists[min_idx]
                    best_target = strokes[j][min_idx]
            
            # Nếu điểm hở cách nhau dưới 5 pixel (khe hở do DFS để lại), 
            # ép tọa độ của nó trùng khớp 100% với điểm trên nét kia.
            if best_dist <= 5.0:
                strokes[i][end_idx] = best_target
                
    return strokes

def solve_control_points_for_stroke(Q, degree=3, smooth_lambda=0.001):
    m = len(Q)
    
    # THUẬT TOÁN SCALE ĐỘNG (BÍ MẬT CỦA THẦY): 
    # Nét càng dài, số Control Points càng lớn. Giới hạn tối đa là 80.
    n = min(80, max(degree + 1, m // 3 + 2))
    
    ubar = chord_length_parameterization(Q)
    U = generate_knot_vector(n, degree)
    
    N = np.zeros((m, n))
    for k in range(m):
        for i in range(n):
            N[k, i] = Nip(i, degree, ubar[k], U)
            
    NT_N = np.dot(N.T, N)
    NT_Q = np.dot(N.T, Q)
    A = NT_N + smooth_lambda * np.eye(n)
    P = np.linalg.lstsq(A, NT_Q, rcond=None)[0]
    
    return P, U

def export_dutmod_multiple_curves(filepath, curves, degree):
    """
    Xuất file chuẩn DUTMod/DISCO với hỗ trợ nhiều đường B-spline (Multi-curves).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, (P, U) in enumerate(curves):
            f.write("==========================\n\n")
            f.write("[BSPLINECURVE]\n\n")
            f.write(f"{len(P)}, {degree}, 1 // UNum, UDegree, UKnotType (Curve {idx+1})\n\n")

            f.write("// Control Points\n")
            for p in P:
                x, y, z = p
                f.write(f"{x:.2f} {y:.2f} {z:.2f} 1.00000000 0\n")

            f.write("\n// UKnot\n")
            for u in U:
                f.write(f"{u:.8f}\n")
            f.write("\n")

if __name__ == "__main__":
    input_file = "data/diempixel.dat"
    output_file = "data/bsplinecurve.dat"
    degree = 3
    
    # Thông số tối giản (không cần max_ctrl cố định nữa vì hàm đã tự tính)
    smooth_lambda = 0.001 
    jump_threshold = 2.0  

    print("=== MODULE KHANG: MULTI-STROKE DYNAMIC CONTROL POINTS ===")
    
    strokes = read_and_split_strokes(input_file, jump_threshold)
    print(f"-> Phat hien {len(strokes)} net rieng biet.")
    
    curves = []
    for i, stroke_points in enumerate(strokes):
        P, U = solve_control_points_for_stroke(stroke_points, degree, smooth_lambda)
        curves.append((P, U))
        
    export_dutmod_multiple_curves(output_file, curves, degree)
    print(f"-> Vi tri file: {output_file}")