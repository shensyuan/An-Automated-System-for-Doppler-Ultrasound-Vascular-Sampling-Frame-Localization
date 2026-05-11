import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV3Large
from keras.optimizers import Adam
import keras.backend as K

from post_processor import post_process
import pre_processor
import time
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNetV3Large
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.3 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.7 * dice_loss(y_true, y_pred)

def attention_gate(x, skip, filters):
    g = layers.Conv2D(filters, (1, 1), padding='same')(x)
    s = layers.Conv2D(filters, (1, 1), padding='same')(skip)
    attn = layers.Add()([g, s])
    attn = layers.Activation('relu')(attn)
    attn = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attn)
    return layers.Multiply()([skip, attn])

def build_mobilenetv3_unet(input_shape=(224, 224, 1), use_attention=False):
    """use_attention=True 用新架構，False 用舊架構"""
    inputs = layers.Input(shape=input_shape)
    x3 = layers.Concatenate()([inputs, inputs, inputs])
    encoder = MobileNetV3Large(input_tensor=x3, include_top=False, weights=None, include_preprocessing=False)
    s1 = encoder.get_layer('re_lu').output
    s2 = encoder.get_layer('expanded_conv_2_add').output
    s3 = encoder.get_layer('expanded_conv_5_add').output
    s4 = encoder.get_layer('expanded_conv_11_add').output
    bridge = encoder.output

    def upsample_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        if use_attention:
            skip = attention_gate(x, skip, filters // 4)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    d1 = upsample_block(bridge, s4, 256)
    d2 = upsample_block(d1, s3, 128)
    d3 = upsample_block(d2, s2, 64)
    d4 = upsample_block(d3, s1, 32)
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    name = "MobileNetV3_AttUNET" if use_attention else "MobileNetV3_UNET"
    model = models.Model(inputs, outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=combined_loss,
                  metrics=['accuracy', dice_coef, iou_coef])
    return model


def load_test_data(test_dir, frames, target_size=(224, 224)):
    mask_dir = os.path.join(test_dir, "test_masks_3")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError("cannot find test_data/test_images or test_data/test_masks")

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
    Y = masks.astype(np.float32) / 255.0
    return np.expand_dims(X, -1), np.expand_dims(Y, -1)


def dice_per_image(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # C:\collega\Project\data\train\test_data\test_masks
    BASE_DIR   = "C:\\collega\\Project\\data\\train"
    VIDEO_DIR  = os.path.join(BASE_DIR, "input_data/data3")
    TEST_DIR   = os.path.join(BASE_DIR, "test_data")
    LINE_DIR   = os.path.join(TEST_DIR, "line")
    MODEL_PATH = os.path.join(BASE_DIR, "model/vessel_lumen_mobilenet_large_unet.h5")
    PRED_DIR   = os.path.join(BASE_DIR, "pred_mask")
    PRED_DIR   = os.path.join(PRED_DIR, "predictions_output_large")
    OUTPUT_DIR   = os.path.join(BASE_DIR, "result/line_large")
    
    frames ={}
    # filenames = []
    # all_line_frames = []
    # all_frames = []
    
    pre_start_time = time.perf_counter()
    for filename in os.listdir(VIDEO_DIR):
        # output_folder = os.path.join(TEST_DIR, "test_images")
        # output_line_folder = LINE_DIR
        # os.makedirs(output_folder, exist_ok=True)
        # os.makedirs(output_line_folder, exist_ok=True)  
        video_path = os.path.join(VIDEO_DIR, filename)
        no_ui_frames, line_frames, file_names, image_h, image_w = pre_processor.extract_frames_with_timestamp(video_path, output_name="data3_" + filename.split(".")[0])
        
        # filenames.extend(file_names)
        # all_line_frames.extend(line_frames)
        # all_frames.extend(no_ui_frames)
        
        for img, line, fname in zip(no_ui_frames, line_frames, file_names):
            frames[fname] = {
                "img": img,
                "line": line,
                "file_name": fname
            }
        
    pre_end_time = time.perf_counter()
    pre_elapsed_time = pre_end_time - pre_start_time
    
    print("reading testing data...")
    raw_imgs, raw_masks, filenames = load_test_data(TEST_DIR, frames)    
    print(f"total {len(raw_masks)} images")

    X_test, Y_test = preprocess_for_inference(raw_imgs, raw_masks)

    print(f"loading model {MODEL_PATH} ...")
    model = build_mobilenetv3_unet(input_shape=(224, 224, 1), use_attention=False)
    model.load_weights(MODEL_PATH)

    print("\ngenerating predicted mask...")
    model_start_time = time.perf_counter()
    
    predictions = model.predict(X_test, verbose=0)
    
    model_end_time = time.perf_counter()
    model_elapsed_time = model_end_time - model_start_time

    os.makedirs(PRED_DIR, exist_ok=True)
    result_path = os.path.join(BASE_DIR, "result/test_mobilenet_large_results.txt")
    dice_scores = []
    masks = []

    with open(result_path, "w", encoding="utf-8") as f:
        header = f"{'Filename':<40} {'Dice':>8}\n" + "-" * 55 + "\n"
        f.write(header)
        print(header, end="")

        for fname, pred, gt in zip(filenames, predictions, Y_test):
            d = dice_per_image(gt, pred)
            dice_scores.append(d)
            line = f"{fname:<40} {d:>8.4f}\n"
            f.write(line)
            print(line, end="")

            mask = (pred > 0.5).astype(np.uint8) * 255
            mask = np.squeeze(mask)
            frames[fname]["pred"] = mask
            # cv2.imencode('.png', mask)[1].tofile(os.path.join(PRED_DIR, fname))

        dice_scores = np.array(dice_scores)
        stats = (
            "-" * 55 + "\n"
            f"Mean ± Std : {dice_scores.mean():.4f} ± {dice_scores.std():.4f}\n"
            f"Min / Max  : {dice_scores.min():.4f} / {dice_scores.max():.4f}\n"
            f"Count      : {len(dice_scores)}\n"
            f"Masks dir  : {PRED_DIR}\n"
        )
        f.write(stats)
        print(stats)
        
        
    post_start_time = time.perf_counter()
        
    post_process(frames,image_h, image_w, OUTPUT_DIR)
    
    post_end_time = time.perf_counter()
    post_elapsed_time = post_end_time - post_start_time

    print(f"Results saved to {result_path}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"times : {elapsed_time:.2f} s")
    print(f"Pre-processing elapsed time: {pre_elapsed_time:.2f} s")
    print(f"Model elapsed time: {model_elapsed_time:.2f} s")
    print(f"Post-processing elapsed time: {post_elapsed_time:.2f} s")
