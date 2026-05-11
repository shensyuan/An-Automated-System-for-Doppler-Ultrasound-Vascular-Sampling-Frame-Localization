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

def resize(img, size):

    h, w = img.shape[:2]

    # 計算縮放比例
    scale = size / min(h, w)

    if scale > 1.0:
        scale = 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize
    resized = cv2.resize(img, (new_w, new_h))

    return resized

def center_crop(img, size):

    h, w = img.shape[:2]

    # size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    crop = img[start_y:start_y+size, start_x:start_x+size]

    return crop

def save_and_classify_frame(original_frame):

    h, w = original_frame.shape[:2]

    # 只抓右上角區域
    roi = original_frame[0:int(h*0.15), int(w*0.75):w]

    if roi.size == 0:
        return

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ========= 1. 偵測白色文字 =========
    _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(white_mask)

    # ========= 2. 偵測黑色背景 =========
    _, black_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    black_pixels = cv2.countNonZero(black_mask)

    # ========= 3. 找文字輪廓 =========
    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    text_like_contours = 0

    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)

        # 過濾太小的 noise
        if 5 < wc < 80 and 5 < hc < 40:
            text_like_contours += 1

    # ========= Debug =========
    # print(frame_name, white_pixels, black_pixels, text_like_contours)

    # ========= 判斷 UI =========
    if (
        white_pixels > 80 and
        black_pixels > 300 and
        text_like_contours >= 2
    ):
        return "with_ui"
    else:
        return "no_ui"

def detected_line(img, filename):

    # ====== 建立彩色範圍 ======
    lower_green = np.array([30, 130, 30])
    upper_green = np.array([95, 255, 255])

    # ====== 抓取符合色彩範圍內的像素成為新影像 ======
    mask = cv2.inRange(img, lower_green, upper_green) 

    mask = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # ====== 形態學處理 ====== 
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # ====== 偵測所有線段 ====== 
    # rho=1, theta=1度, threshold=20 (因為片段可能很短，門檻設低一點)
    lines = cv2.HoughLinesP(
        mask_dilated, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=10, 
        maxLineGap=50  # 這裡設大一點，讓 OpenCV 嘗試自己連線
    )

    if lines is not None:
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.append((x1, y1))
            all_points.append((x2, y2))

        #  ====== 關鍵步驟：尋找「最遠兩端點」：找 Y 座標最小（最上方）與最大（最下方）的點 ====== 
        p_top = min(all_points, key=lambda p: p[1])
        p_bottom = max(all_points, key=lambda p: p[1])

        # ====== 計算長直線資訊 ====== 
        x1, y1 = p_top
        x2, y2 = p_bottom
        
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # ====== 繪製結果 ====== 
        # result_img = img_ori.copy()
        result_img = np.zeros_like(img)
        cv2.line(result_img, p_top, p_bottom, (0, 0, 255), 2) 
        # cv2.circle(result_img, p_top, 5, (0, 255, 0), -1)
        # cv2.circle(result_img, p_bottom, 5, (0, 255, 0), -1) 

        # print(f"偵測完成！")
        # print(f"頂點: {p_top}, 底點: {p_bottom}")
        # print(f"連線總長度: {length:.2f}")
        # print(f"連線角度: {angle:.2f}")

        return result_img
    else:
        print(f"{filename} 未能偵測到線段")
        
    return img

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
        
        image_h, image_w = 700, 900
        
        line = detected_line(frame, filename)
        result = remove_green_red(frame)
        result = resize_with_padding(result, 224)
        enhanced = apply_clahe(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))         

        # 儲存圖片
        statue = save_and_classify_frame(frame)
        if statue == "no_ui":
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
