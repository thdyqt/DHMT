import cv2
import numpy as np
import os
import math


def trace_segment(start_pt, points):
    segment = [start_pt]
    current = start_pt

    while True:
        neighbors = sorted(
            [p for p in points if math.dist(current, p) <= 1.5],
            key=lambda p: math.dist(current, p)
        )

        if not neighbors:
            break

        next_pt = neighbors[0]
        segment.append(next_pt)
        points.remove(next_pt)
        current = next_pt

    return segment


def extract_smooth_skeleton(skel):
    """
    Thuật toán dò vết chia nhánh: Mỗi nhánh sẽ được tách rời hoàn toàn.
    Chống đè nét (Overlapping) 100%.
    """
    y_idx, x_idx = np.where(skel == 255)
    points = set(zip(x_idx, y_idx))
    ordered_points = []
    
    while points:
        # 1. Ưu tiên bắt đầu dò từ các điểm đầu mút (end-point có 1 lân cận)
        start_p = next(iter(points))
        for p in points:
            neighbors = sum(1 for dx in [-1,0,1] for dy in [-1,0,1] 
                            if (dx!=0 or dy!=0) and (p[0]+dx, p[1]+dy) in points)
            if neighbors == 1:
                start_p = p
                break
                
        # 2. Bắt đầu rút nét
        current = start_p
        points.remove(current)
        ordered_points.append(current)
        
        while True:
            # Tìm các pixel nối liền kế bên
            neighbors = [(current[0]+dx, current[1]+dy) 
                         for dx in [-1,0,1] for dy in [-1,0,1] 
                         if (dx!=0 or dy!=0) and (current[0]+dx, current[1]+dy) in points]
            
            if not neighbors:
                break # Hết đường, ngắt nét tại đây.
                
            # Đi tới pixel liền kề đầu tiên và xóa nó đi (để không bao giờ đi lùi)
            current = neighbors[0]
            points.remove(current)
            ordered_points.append(current)
            
    return ordered_points

def process_image():
    print("=== MODULE 1: XU LY ANH CHU KY ===")
    print("Thu muc hien tai:", os.getcwd())

    while True:
        image_path = input("Nhap duong dan anh (hoac 'exit'): ").strip()

        if image_path.lower() == "exit":
            print("Da thoat chuong trinh xu ly anh.")
            break

        if not os.path.exists(image_path):
            print("Khong tim thay file anh:", image_path)
            continue

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Khong doc duoc anh:", image_path)
            continue

        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        closed = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        thick_binary = cv2.dilate(
            closed,
            kernel,
            iterations=1
        )

        skel = cv2.ximgproc.thinning(
            thick_binary,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )

        final_smoothed_points = extract_smooth_skeleton(skel)

        if len(final_smoothed_points) == 0:
            print("Khong tach duoc diem nao tu anh.")
            continue

        os.makedirs("data", exist_ok=True)

        output_path = os.path.join("data", "diempixel.dat")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{len(final_smoothed_points)}\n")

            for x, y in final_smoothed_points:
                f.write(
                    f"{float(x):.2f} "
                    f"{float(img.shape[0] - y):.2f} "
                    f"0.00 1.00\n"
                )

        print(f"Da xuat {len(final_smoothed_points)} diem ra file:")
        print(output_path)

        result_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result_vis[skel == 255] = [255, 200, 200]

        for pt in final_smoothed_points:
            cv2.circle(result_vis, pt, 2, (0, 255, 0), -1)

        cv2.imshow("1. Anh Goc", img)
        cv2.imshow("2. Net Manh Skeleton", skel)
        cv2.imshow("3. Diem sau khi truy vet", result_vis)

        print("Nhan phim bat ky tren cua so anh de tiep tuc...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_image()