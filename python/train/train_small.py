import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import MobileNetV3Small

from keras.models import load_model

from post_processor import post_process
import pre_processor
import time


# =========================
# Metrics
# =========================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# =========================
# Model
# =========================
def build_mobilenetv3_unet(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)

    x3 = layers.Concatenate()([inputs, inputs, inputs])
    encoder = MobileNetV3Small(
        input_tensor=x3,
        include_top=False,
        weights="imagenet"
    )

    s1 = encoder.layers[4].output
    s2 = encoder.get_layer("expanded_conv_project_bn").output
    s3 = encoder.get_layer("expanded_conv_2_add").output
    s4 = encoder.get_layer("expanded_conv_5_add").output
    bridge = encoder.output

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding="same"
        )(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1, s3, 128)
    d3 = upsample_block(d2, s2, 64)
    d4 = upsample_block(d3, s1, 32)

    x = layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding="same"
    )(d4)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    model = models.Model(inputs, outputs, name="MobileNetV3_UNET")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[dice_coef, iou_coef]
    )
    return model


# =========================
# Data loading
# =========================
def load_test_data(test_dir, frames, target_size=(224, 224)):
    mask_dir = os.path.join(test_dir, "test_masks_1")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"cannot find mask directory: {mask_dir}")

    for fname in frames.keys():
        base_name = fname.rsplit('.', 1)[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_label.png")

        if os.path.exists(mask_path):
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != target_size: mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            frames[fname]["raw_mask"] = mask
                
        else:
            print(f"⚠️ 找不到對應標記圖，跳過: {fname}")
            
    images = []
    masks = []
    filenames = []

    for fname, d in frames.items():

        if "raw_mask" not in d:
            continue

        images.append(d["img"])
        masks.append(d["raw_mask"])
        filenames.append(fname)

    return np.array(images), np.array(masks), filenames


def preprocess_for_inference(imgs, masks):
    X = (imgs.astype(np.float32) / 255.0 - 0.5) / 0.5

    # binary mask: 0 or 1
    Y = (masks > 127).astype(np.float32)

    return np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1)


def calc_per_image_metrics(y_true, y_pred, smooth=1e-6):
    dice_list = []
    iou_list = []

    for i in range(len(y_true)):
        gt = y_true[i].flatten().astype(np.float32)
        pred = y_pred[i].flatten().astype(np.float32)

        intersection = np.sum(gt * pred)
        gt_sum = np.sum(gt)
        pred_sum = np.sum(pred)

        dice = (2.0 * intersection + smooth) / (gt_sum + pred_sum + smooth)
        iou = (intersection + smooth) / (gt_sum + pred_sum - intersection + smooth)

        dice_list.append(dice)
        iou_list.append(iou)

    return np.array(dice_list), np.array(iou_list)


# =========================
# Main
# =========================
if __name__ == "__main__":
    total_start_time = time.perf_counter()
    
    BASE_DIR   = "C:\\collega\\Project\\data\\train"
    MODEL_PATH = os.path.join(BASE_DIR, "model/mobilenet_small_unet_last_train237_v456_test1.h5")
    TEST_DIR   = os.path.join(BASE_DIR, "test_data")
    PRED_DIR = os.path.join(BASE_DIR, "pred_mask/predictions_output_small_last_train237_v456_test1_noearly")
    VIDEO_DIR  = os.path.join(BASE_DIR, "input_data/data1")
    OUTPUT_DIR   = os.path.join(BASE_DIR, "result/line_small")

    os.makedirs(PRED_DIR, exist_ok=True)

    print("Using CPU only.")
    
    pre_start_time = time.perf_counter()
    frames ={}
    for filename in os.listdir(VIDEO_DIR):
        video_path = os.path.join(VIDEO_DIR, filename)
        no_ui_frames, line_frames, file_names, image_h, image_w = pre_processor.extract_frames_with_timestamp(video_path, output_name="data1_" + filename.split(".")[0])
        
        for img, line, fname in zip(no_ui_frames, line_frames, file_names):
            frames[fname] = {
                "img": img,
                "line": line,
                "file_name": fname
            }
    pre_end_time = time.perf_counter()
    pre_elapsed_time = pre_end_time - pre_start_time
    
    print("reading testing data...")

    raw_imgs, raw_masks, filenames = load_test_data(
        TEST_DIR,
        frames
    )

    print(f"total {len(raw_imgs)} images")

    if len(raw_imgs) == 0:
        raise ValueError("沒有讀到任何測試圖片，請檢查 images/masks 路徑和檔名。")

    X_test, Y_test = preprocess_for_inference(raw_imgs, raw_masks)

    print(f"building model and loading weights from {MODEL_PATH} ...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1))
    model.load_weights(MODEL_PATH)

    # Warm-up: not included in timing
    print("\nWarming up...")
    _ = model.predict(X_test[:min(2, len(X_test))], verbose=0)

    # Only measure inference time
    print("\nTiming inference...")
    model_start_time = time.perf_counter()
    predictions = model.predict(X_test, verbose=0)
    model_end_time = time.perf_counter()
    model_elapsed_time = model_end_time - model_start_time

    total_time = model_end_time - model_start_time
    per_image_time = total_time / len(X_test)

    pred_binary = (predictions > 0.5).astype(np.float32)

    dice_scores, iou_scores = calc_per_image_metrics(Y_test, pred_binary)
    

    print("\n===== Final Test Result =====")

    print(
        f"Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f} "
        f"(min={np.min(dice_scores):.4f}, max={np.max(dice_scores):.4f})"
    )

    print(
        f"IoU : {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f} "
        f"(min={np.min(iou_scores):.4f}, max={np.max(iou_scores):.4f})"
    )

    print("\nInference Time")
    print(f"Total inference time: {total_time:.4f} sec")
    print(f"Mean time per image : {per_image_time:.6f} sec")
    print(f"FPS                 : {1.0 / per_image_time:.2f}")

    print("\nGenerating predicted masks...")
    # for i in range(len(raw_imgs)):
    #     filename = filenames[i]
    #     pred_mask = np.squeeze(pred_binary[i]).astype(np.uint8) * 255

    #     save_path = os.path.join(PRED_DIR, filename)
    #     cv2.imwrite(save_path, pred_mask)
        
    for fname, pred, gt in zip(filenames, pred_binary, Y_test):
            mask = (pred > 0.5).astype(np.uint8) * 255
            mask = np.squeeze(mask)
            frames[fname]["pred"] = mask
            # cv2.imencode('.png', mask)[1].tofile(os.path.join(PRED_DIR, fname))
            
    post_start_time = time.perf_counter()
        
    post_process(frames,image_h, image_w, OUTPUT_DIR)
    
    post_end_time = time.perf_counter()
    post_elapsed_time = post_end_time - post_start_time

    total_end_time = time.perf_counter()
    elapsed_time = total_end_time - total_start_time
    print(f"times : {elapsed_time:.2f} s")
    print(f"Pre-processing elapsed time: {pre_elapsed_time:.2f} s")
    print(f"Model elapsed time: {model_elapsed_time:.2f} s")
    print(f"Post-processing elapsed time: {post_elapsed_time:.2f} s")
