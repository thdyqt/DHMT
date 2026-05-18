import cv2
import numpy as np
import os
import math

def extract_smooth_skeleton(skel):
    y_idx, x_idx = np.where(skel == 255)
    points = set(zip(x_idx, y_idx))
    
    junctions = set()
    for p in points:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                if (p[0]+dx, p[1]+dy) in points:
                    neighbors += 1
        if neighbors >= 3:
            junctions.add(p)

    broken_points = points - junctions
    
    segments = []
    while broken_points:
        start_pt = broken_points.pop()
        segment = [start_pt]
        current = start_pt
        
        while True:
            neighbors = [p for p in broken_points if math.dist(current, p) <= 1.5]
            if neighbors:
                next_pt = neighbors[0]
                segment.append(next_pt)
                broken_points.remove(next_pt)
                current = next_pt
            else:
                break
                
        if len(segment) > 5:
            segments.append(segment)

    final_points = []
    epsilon = 2.0
    for seg in segments:
        seg_arr = np.array(seg)
        smoothed_seg = cv2.approxPolyDP(seg_arr, epsilon, closed=False)
        for pt in smoothed_seg:
            final_points.append(tuple(pt[0]))
            
    final_points.sort(key=lambda p: p[0])
    
    return final_points

def process_image():
    while True:
        image_path = input("Nhap duong dan anh (hoac 'exit'): ").strip()
        if image_path.lower() == 'exit': break
        if not os.path.exists(image_path): continue

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        thick_binary = cv2.dilate(closed, kernel, iterations=1)

        skel = cv2.ximgproc.thinning(thick_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        final_smoothed_points = extract_smooth_skeleton(skel)

        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("data", "diempixel.dat")
        with open(output_path, "w") as f:
            f.write(f"{len(final_smoothed_points)}\n")
            for x, y in final_smoothed_points:
                f.write(f"{float(x):.2f} {float(img.shape[0] - y):.2f} 0.00 1.00\n")

        result_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result_vis[skel == 255] = [255, 200, 200] 
        
        for pt in final_smoothed_points:
            cv2.circle(result_vis, pt, 2, (0, 255, 0), -1)

        cv2.imshow("1. Anh Goc", img)
        cv2.imshow("2. Net Manh", skel)
        cv2.imshow("3. Do thi Diem RDP", result_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image()