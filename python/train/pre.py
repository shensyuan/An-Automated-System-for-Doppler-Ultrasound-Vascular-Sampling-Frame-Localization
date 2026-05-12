import cv2
import os
import math
import numpy as np
from matplotlib.pyplot import hsv

def apply_clahe(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    return enhanced

def remove_green_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ========= 綠色 =========
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # ========= 紅色 (兩段) =========
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合併綠色和紅色的 mask
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_green, mask_red)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    return result

def resize_with_padding(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    # padding
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return padded

def detected_line(img):
    # ====== 建立彩色範圍 ======
    lower_green = np.array([30, 130, 30])
    upper_green = np.array([95, 255, 255])

    # ====== 抓取符合色彩範圍內的像素成為新影像 ======
    mask = cv2.inRange(img, lower_green, upper_green) 
    
    green_points = np.where(mask == 255)
    
    if len(green_points[0]) > 0:
        # 將座標轉換成 (x, y) 的列表
        all_points = list(zip(green_points[1], green_points[0]))
        
        # 找出最上方的點（Y最小）和最下方的點（Y最大）
        p_top = min(all_points, key=lambda p: p[1])
        p_bottom = max(all_points, key=lambda p: p[1])
        
        dx = p_bottom[0] - p_top[0]
        dy = p_bottom[1] - p_top[1]
        
        # 銳角（弧度）
        alpha_rad = math.atan(abs(dx / dy))
        # 轉換為度
        alpha_deg = math.degrees(alpha_rad)
        
        # 決定正負
        if dx * dy < 0:  
            return alpha_deg, p_top[0]
        else:            
            return -alpha_deg, p_top[0]
    
    return None, None

def get_line_point(top_x, top_y, angle_deg, length):
    """
    根據最上方端點、角度和長度，計算最底部端點的座標
    """
    top_x = float(top_x)
    top_y = float(top_y)
    angle_rad = math.radians(float(angle_deg))
    length = float(length)
    
    dx = math.sin(angle_rad) * length
    dy = math.cos(angle_rad) * length
    
    bottom_x = top_x + dx
    bottom_y = top_y + dy
    
    return int(round(bottom_x)), int(round(bottom_y))

def extract_frames_with_timestamp(input_dir, output_name=None):

    # ======= 遍歷資料夾 =======
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]

    print(f"找到 {len(image_files)} 張 Mask 圖片，開始對應原圖處理...")
    
    file_names = []
    no_ui_frames = []
    line_frames = []
    

    for filename in image_files:
        # ======= 讀取圖像 ======= 
        img_path = os.path.join(input_dir, filename)
        frame = cv2.imread(img_path)
        
        image_h, image_w = frame.shape[:2]
        
        line = [(0, 0), (0, 0)]
        line_angel, top_x = detected_line(frame)
        print(f"{filename} 角度: {line_angel}, top_x: {top_x}")
        
        if line_angel is None or top_x is None:
            print(f"{filename} 未能偵測到線段，跳過")
        else:
            line[0] = (top_x, 0)
            line[1] = get_line_point(line[0][0], line[0][1], line_angel, image_h)

        result = remove_green_red(frame)
        result = resize_with_padding(result, 224)
        enhanced = apply_clahe(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))         

        # 儲存圖片
        # statue = classify_frame(frame)
        # if statue == "no_ui":
        no_ui_frames.append(enhanced)
        line_frames.append(line)
        file_names.append(filename)
            # cv2.imwrite(os.path.join("output/line", filename), line)
            # cv2.imwrite(os.path.join("output/Img", filename), enhanced)
            
    return no_ui_frames, line_frames, file_names, image_h, image_w

# input_folder = "input_data\\input_image\\data3"
# output_folder = "output/Img"
# output_line_folder = "output/line"
# os.makedirs(output_folder, exist_ok=True)
# os.makedirs(output_line_folder, exist_ok=True)
# extract_frames_with_timestamp(input_folder)
# print("處理完成！")
