import math
import cv2
import os
import time
import numpy as np
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt, label
from scipy.interpolate import splprep, splev

def resize(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / min(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    return resized

def crop(img, x1, x2, y1, y2):

    # x1, y1 = 0, 100
    # x2, y2 = 900, 800
    
    cropped = img[y1:y2, x1:x2]

    return cropped

def is_valid_centerline(x_pts, y_pts, img_w=224):
    """ 判定中心線合法性並過濾雜訊 """
    if len(x_pts) < 15: return False
    
    width = np.max(x_pts) - np.min(x_pts)
    y_variability = np.sum(np.abs(np.diff(y_pts))) / (width + 1e-6)
    
    if width < 20: return False 
    if y_variability > 1.5: return False

    return True

def process_single_centerline(img_orig, mask_224):
    """ 處理單一張影像並生成中心線 Mask """
    
    # 讀取影像
    # img_orig = cv2.imread(img_path)
    # mask_224 = cv2.imread(label_path, 0)
    
    # if img_orig is None or mask_224 is None:
    #     print(f"Error: 無法讀取影像或標籤。檢查路徑: \n{img_path}\n{label_path}")
    #     return

    h_orig, w_orig = img_orig.shape[:2]
    scale_x, scale_y = w_orig / 224.0, h_orig / 224.0

    # 建立純黑畫布
    centerline_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # 二值化與標記連通域
    _, binary_mask = cv2.threshold(mask_224, 127, 255, cv2.THRESH_BINARY)
    labeled_array, num_features = label(binary_mask > 0)
    
    for i in range(1, num_features + 1):
        single_region = (labeled_array == i)
        if np.sum(single_region) < 20: continue

        dist_map = distance_transform_edt(single_region)
        skel_region = medial_axis(single_region)
        y_idx, x_idx = np.where(skel_region)
        
        region_best_pts = {}
        for x, y in zip(x_idx, y_idx):
            val = dist_map[y, x]
            if x not in region_best_pts or val > region_best_pts[x][1]:
                region_best_pts[x] = (y, val)
        
        sorted_x = np.array(sorted(region_best_pts.keys()))
        sorted_y = np.array([region_best_pts[x][0] for x in sorted_x])

        if not is_valid_centerline(sorted_x, sorted_y):
            continue

        # 座標縮放
        pts_scaled = np.column_stack((sorted_x * scale_x, sorted_y * scale_y))

        try:
            k_val = min(3, len(pts_scaled)-1)
            tck, u = splprep([pts_scaled[:, 0], pts_scaled[:, 1]], s=len(pts_scaled)*12, k=k_val)
            u_fine = np.linspace(0, 1, 2000) 

            x_fine, y_fine = splev(u_fine, tck)
            
            # 物理級細線繪製
            for x, y in zip(x_fine, y_fine):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w_orig and 0 <= iy < h_orig:
                    centerline_mask[iy, ix] = 255
        except Exception as e:
            print(f"Spline fitting error: {e}")
            continue

    # 儲存結果
    # cv2.imwrite(output_path, centerline_mask)
    # print(f"成功儲存中心線 Mask 至: {output_path}")
    return centerline_mask

def draw_perpendicular_line(image, line_p1, line_p2, point, length=50, color=(0, 255, 0), thickness=2):
    """
    在指定點畫出與直線垂直的線
    
    Parameters:
        image: 輸入圖像
        line_p1: 原直線的第一個點 (x, y)
        line_p2: 原直線的第二個點 (x, y)
        point: 要畫垂直線的點 (x, y)
        length: 垂直線的長度（兩側各 length/2）
        color: 線條顏色 (B, G, R)
        thickness: 線條粗細
    """
    # 計算原直線的方向向量
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    
    # 計算原直線的長度
    line_length = math.sqrt(dx**2 + dy**2)
    
    if line_length == 0:
        return None
    
    # 計算單位方向向量
    ux = dx / line_length
    uy = dy / line_length
    
    # 垂直向量（旋轉90度）
    perp_x = -uy
    perp_y = ux
    
    # 計算垂直線的兩個端點
    half_length = length / 2
    p1 = (int(point[0] + perp_x * half_length), int(point[1] + perp_y * half_length))
    p2 = (int(point[0] - perp_x * half_length), int(point[1] - perp_y * half_length))
    
    # 繪製垂直線
    cv2.line(image, p1, p2, color, thickness)
    
    return p1, p2

def find_RangeGate(start_pt, target_pt, img):
    """
    沿著 start_pt->p_top 和 start_pt->p_bottom 尋找最近的黑白交界
    :param start_pt: 起始點 (x, y)
    :param target_pt: 線段一端 (x, y)
    :param img: 二值化影像 (255為血管內部, 0為背景)
    :return: (top_intersection, bottom_intersection)
    """
    h, w = img.shape
    start_x, start_y = start_pt
    target_x, target_y = target_pt
        
    # 1. 計算方向向量
    dx = target_x - start_x
    dy = target_y - start_y
    distance = np.sqrt(dx**2 + dy**2)
        
    if distance == 0:
        return start_pt
        
    # 2. 單位向量 (每次移動 1 像素)
    ux = dx / distance
    uy = dy / distance
        
    # 3. 沿線搜尋
    last_pt = start_pt
    for d in np.arange(0, distance, 1.0):
        curr_x = int(start_x + d * ux) 
        curr_y = int(start_y + d * uy)
            
        # 檢查是否超出影像邊界
        if not (0 <= curr_y < h and 0 <= curr_x < w):
            return last_pt
            
        # 偵測交界點
        if img[curr_y, curr_x] == 0:
            final_x = int(round(curr_x - ux * 2))
            final_y = int(round(curr_y - uy * 2))
            
            # 檢查是否還在影像內
            final_x = max(0, min(w - 1, final_x))
            final_y = max(0, min(h - 1, final_y))
            return (final_x, final_y)
            
        last_pt = (curr_x, curr_y)
            
    return last_pt

def get_boundary_intersection_direct(mask_shape, center_pt, angle_deg):
    """
    計算指定角度射線與圖片邊界的交點
    :param mask_shape: (height, width)
    :param center_pt: (x, y)
    :param angle_deg: 絕對角度 (度)
    :return: (pt_far_1, pt_far_2) 兩個在圖片邊緣的點
    """
    h, w = mask_shape
    cx, cy = center_pt
    
    # 1. 取得方向向量
    rad = np.radians(angle_deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    
    # 2. 計算到達四個邊界需要的距離 (d = delta_pos / direction)
    distances = []
    
    # 避免除以零
    epsilon = 1e-9
    
    # 檢查與垂直邊界相交 (左右)
    if abs(dx) > epsilon:
        distances.append((0 - cx) / dx)      # 左邊界
        distances.append((w - 1 - cx) / dx)  # 右邊界
        
    # 檢查與水平邊界相交 (上下)
    if abs(dy) > epsilon:
        distances.append((0 - cy) / dy)      # 上邊界
        distances.append((h - 1 - cy) / dy)  # 下邊界
        
    # 3. 找出正向最靠近的距離，以及負向最靠近的距離
    t_pos = min([t for t in distances if t > 0])
    t_neg = max([t for t in distances if t < 0])
    
    # 4. 根據距離回推座標
    pt_edge_1 = (int(cx + t_pos * dx), int(cy + t_pos * dy))
    pt_edge_2 = (int(cx + t_neg * dx), int(cy + t_neg * dy))
    
    if pt_edge_1[1] > pt_edge_2[1]:
        p_bottom = pt_edge_1
        p_top = pt_edge_2
    else:
        p_bottom = pt_edge_2
        p_top = pt_edge_1
    
    return p_top, p_bottom

def get_absolute_angle(v, relative_angle_deg):
    """
    計算旋轉後的向量相對於圖片水平軸(X軸)的絕對角度
    :param v: 基準向量 (dx, dy)
    :param relative_angle_deg: 相對於基準向量的夾角 (度)
    :return: 絕對角度 (0~360 度)
    """
    # 1. 取得基準向量本身的絕對角度 (弧度)
    # np.arctan2 參數順序是 (y, x)
    base_angle_rad = np.arctan2(v[1], v[0])
    
    # 2. 加上相對夾角 (轉為弧度)
    relative_angle_rad = np.radians(relative_angle_deg)
    final_angle_rad = base_angle_rad + relative_angle_rad
    
    # 3. 轉回角度制
    final_angle_deg = np.degrees(final_angle_rad)
    
    # 4. 正規化到 0~360 度之間 (選用，方便閱讀)
    final_angle_deg = final_angle_deg % 360
    
    return final_angle_deg

def get_tangent_direction(skeleton, point, window_size=15):
    """
    用PCA獲取骨架上某點的切線方向
    Returns:
        direction: 切線方向向量 (dx, dy)，永遠有效（不會是 None）
    """
    x0, y0 = point
    h, w = skeleton.shape

    # ===== 直接在函式內檢查 =====
    # 檢查1: skeleton 是否有效
    if skeleton is None or skeleton.size == 0:
        print("Warning: skeleton 為空")
        return np.array([1.0, 0.0])  # 直接回傳預設值
    
    # 檢查2: 點是否在範圍內
    if not (0 <= x0 < w and 0 <= y0 < h):
        print(f"Warning: 點 ({x0}, {y0}) 超出圖像範圍")
        return np.array([1.0, 0.0])
    
    # 檢查3: 點是否在骨架上（選用）
    if skeleton[y0, x0] == 0:
        print(f"Warning: 點 ({x0}, {y0}) 不在骨架上")
        # 可以選擇找最近的骨架點，或直接回傳預設值
        return np.array([1.0, 0.0])

    # 取 ROI
    x1 = max(0, x0 - window_size)
    x2 = min(w, x0 + window_size)
    y1 = max(0, y0 - window_size)
    y2 = min(h, y0 + window_size)

    roi = skeleton[y1:y2, x1:x2]

    # 找 skeleton 點
    ys, xs = np.where(roi > 0)

    # 檢查4: 點數是否足夠
    if len(xs) < 2:
        print(f"Warning: 點 {point} 鄰域內只有 {len(xs)} 個骨架點")
        return np.array([1.0, 0.0])

    # 轉回全圖座標
    xs = xs + x1
    ys = ys + y1
    points = np.column_stack((xs, ys)).astype(np.float32)

    try:
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # 檢查5: 是否所有點相同
        if np.all(centered == 0):
            print(f"Warning: 點 {point} 鄰域內所有點重合")
            return np.array([1.0, 0.0])
        
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]
        
        if direction[0] > 0:
            direction = -direction
        
        # 正規化
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        else:
            return np.array([1.0, 0.0])
            
    except Exception as e:
        print(f"Warning: SVD 失敗: {e}")
        return np.array([1.0, 0.0]) 

def calculate_angle_between_vectors(p1, p2, v_given, absolute=False):
    """
    計算兩點(p1, p2)連線向量與給定向量(v_given)之間的夾角
    :param p1: 點1 (x1, y1)
    :param p2: 點2 (x2, y2)
    :param v_given: 給定向量 (dx, dy)
    :param absolute: 若為 True，回傳 0~90 度的最小夾角 (適合算 Range Gate 與血管夾角)
    :return: 角度 (Degree)
    """
    # 1. 算出兩點連線的向量 A
    vec_a = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    # 2. 準備給定向量 B
    vec_b = np.array(v_given)
    
    # 3. 計算長度 (Norm)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # 防止除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # 4. 計算點積並求出 cos(theta)
    dot_product = np.dot(vec_a, vec_b)
    cos_theta = dot_product / (norm_a * norm_b)
    
    # 限制 cos 範圍在 -1 到 1 之間，避免浮點數誤差導致 NaN
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 5. 轉為角度
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    if absolute:
        # 將角度轉為 -90 ~ 90 度
        if angle_deg > 90:
            angle_deg = angle_deg - 180
            
    return angle_deg

def draw_tangent(img, point, direction, length=20):
    x0, y0 = point
    dx, dy = direction

    # 正規化
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    x1 = int(x0 - dx * length)
    y1 = int(y0 - dy * length)
    x2 = int(x0 + dx * length)
    y2 = int(y0 + dy * length)

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def post_process(frames, image_h, image_w, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"總共有 {len(frames)} 張圖片需要後處理...")

    for fname in frames.keys():
        start = time.perf_counter()

        # ===== 路徑 =====
        # line_path = os.path.join(base_line_dir, filename)
        # mask_path = os.path.join(base_mask_dir, filename.replace("_line.png", ".png"))
        # ori_path  = os.path.join(base_ori_dir, filename.replace("_line.png", ".png"))
        # mask_path = os.path.join(base_mask_dir, filename.replace(".png", "_label.png"))
        # ori_path  = os.path.join(base_ori_dir, filename)
    
        # ===== 讀圖 =====
        line = frames[fname].get("line", None)
        mask = frames[fname].get("raw_mask", None)
        img_ori = frames[fname].get("img", None)

        if line is None or mask is None:
            print(f"跳過 {fname}")
            continue

        if img_ori is None:
            print(f"找不到 {fname} 的原圖")
            img_ori = mask.copy()

        #  ======= 224 -> 900*700 ======= 
        if image_h > image_w:
            img_ori = resize(img_ori, image_h)
            image = resize(mask, image_h)
            width_pad = (image_h - image_w) // 2
            image = crop(image, width_pad, width_pad + image_w, 0, image_h)
        else:
            img_ori = resize(img_ori, image_w)
            image = resize(mask, image_w)
            width_pad = (image_w - image_h) // 2
            image = crop(image, 0, image_w, width_pad, width_pad + image_h)

        
        # ======= 生成中心線 =======
        centerLine = process_single_centerline(img_ori, mask)
        if image_h > image_w:
            width_pad = (image_h - image_w) // 2
            img_ori = crop(img_ori, width_pad, width_pad + image_w, 0, image_h)
            centerLine = crop(centerLine, width_pad, width_pad + image_w, 0, image_h)
        else:
            width_pad = (image_w - image_h) // 2
            img_ori = crop(img_ori, 0, image_w, width_pad, width_pad + image_h)
            centerLine = crop(centerLine, 0, image_w, width_pad, width_pad + image_h)
        
        if np.sum(centerLine > 0) == 0:
            print(f"Warning: {fname}Skeleton is empty, skipping this image.")
            continue 
        
        #  ======= 找線的端點 ======= 
        p_top, p_bottom = line[0], line[1]
        if p_top is None or p_bottom is None:
            print(f"{fname} 未能偵測到線段")
            continue

        #  ======= 畫線 ======= 
        lines = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.line(lines, p_top, p_bottom, 255, 1)

        #  ======= 找到中心線與綠線交點 =======
        intersection_mask = cv2.bitwise_and(lines, centerLine)
        points = np.where(intersection_mask == 255)
        points_list = list(zip(points[1], points[0]))

        # ======== 處理交點 ==========
        if len(points_list) >= 1:
            mask = cv2.bitwise_and(lines, image)
            green_points = np.where(mask == 255)
            
            center = points_list[0]
            
            # ======== Angle ========== 
            direction = get_tangent_direction(centerLine, center)

            # ======== range gate ==========
            if len(green_points[0]) > 0:
                all_points = list(zip(green_points[1], green_points[0]))
                
                in_top = min(all_points, key=lambda p: p[1])
                in_bottom = max(all_points, key=lambda p: p[1])
                intersection_top = find_RangeGate(center, in_top, image)
                intersection_bottom = find_RangeGate(center, in_bottom, image)
        
        # ======== 以原圖處理 ========     
        if len(points_list) < 1:
            
            # ======== 找中心線中點 ========
            ys, xs = np.where(centerLine > 0)
    
            if len(xs) < 20:
                return None, None
            
            # 1. 取得中點
            center_x = int(np.median(xs))
            center_y = int(np.median(ys))
            
            distances = (xs - center_x)**2 + (ys - center_y)**2
            nearest_idx = np.argmin(distances)
            center = (int(xs[nearest_idx]), int(ys[nearest_idx]))
            
            # ======== Angle ========
            direction = get_tangent_direction(centerLine, center)
            
            # ======== 預設線段(-75) ========
            p_top, p_bottom = get_boundary_intersection_direct(image.shape, center, get_absolute_angle(direction, 60))
            
            lines = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.line(lines, p_top, p_bottom, 255, 1) 
                    
            mask = cv2.bitwise_and(lines, image)
            green_points = np.where(mask == 255)
    
            # ======== range gate ==========
            if len(green_points[0]) > 0:
                all_points = list(zip(green_points[1], green_points[0]))
                
                in_top = min(all_points, key=lambda p: p[1])
                in_bottom = max(all_points, key=lambda p: p[1])
                intersection_top = find_RangeGate(center, in_top, image)
                intersection_bottom = find_RangeGate(center, in_bottom, image)

            
        #  ======= 視覺化 =======
        result = img_ori.copy()
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # 畫骨架
        skeleton_color = cv2.cvtColor(centerLine, cv2.COLOR_GRAY2BGR)
        skeleton_color[centerLine == 255] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.7, skeleton_color, 0.3, 0)
        
        # 畫 Range Gate
        draw_perpendicular_line(result, p_top, p_bottom, intersection_top, length=50, color=(0, 0, 255), thickness=2)
        draw_perpendicular_line(result, p_top, p_bottom, intersection_bottom, length=50, color=(0, 0, 255), thickness=2)
        
        # 畫 Angle
        if direction is not None:
            draw_tangent(result, center, direction)
        
        # 畫 beam
        cv2.line(result, p_top, intersection_top, (0, 255, 0), 1)
        cv2.line(result, intersection_bottom, p_bottom, (0, 255, 0), 1)
        angle = calculate_angle_between_vectors(intersection_bottom, intersection_top, direction, absolute=True)
        angle_abs = calculate_angle_between_vectors((0,0), (0,1), direction, absolute=True)

        # ===== 存檔 =====
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, result)
        
        # print(f"{filename} | {time.perf_counter()-start:.3f}s | intersection: {len(points_list)}")
        print(f"{fname} | Angle: {angle_abs:.2f} | 夾角: {angle:.2f} | center: {center}")
        # print(f"p: {p_top}, {p_bottom} | intersection: {intersection_top}, {intersection_bottom} | center: {center} | len(points_list): {len(points_list)}")

    print("\n=== 全部處理完成 ===")
