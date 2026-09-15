/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.superresolution;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.WorkerThread;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.view.ViewGroup;
import android.util.DisplayMetrics;

/**
 * 都卜勒血管取樣框定位 Demo 的主畫面。
 * 載入 assets 內的 TFLite 分割模型，讓使用者挑選一張測試影格後，透過 JNI 依序執行
 * 取樣線偵測 → 去除紅綠疊圖 → native pipeline（前處理 / 模型推論 / 後處理），
 * 最後把視覺化影像與量測數值顯示在畫面上。
 */
public class MainActivity extends AppCompatActivity {
  static {
    System.loadLibrary("SuperResolution");
  }
  private static final String TAG = "SuperResolution";
  private static final String MODEL_NAME = "ESRGAN.tflite";
  private static final int LR_IMAGE_HEIGHT = 700;
  private static final int LR_IMAGE_WIDTH = 900;
  private static final int SR_IMAGE_HEIGHT = 700;
  private static final int SR_IMAGE_WIDTH = 900;
  // 測試用影像
  private static final String LR_IMG_1 = "data1_clip_141s_5.83s.png";//"n1.png"; 
  private static final String LR_IMG_2 = "data1_clip_560s_6.81s.png";//"n2.png";
  private static final String LR_IMG_3 = "data6_0.95s.png";//"n3.png";

  private MappedByteBuffer model;
  private long superResolutionNativeHandle = 0;
  private Bitmap selectedLRBitmap = null;

  private ImageView lowResImageView1;
  private ImageView lowResImageView2;
  private ImageView lowResImageView3;
  private TextView selectedImageTextView;

  /**
   * 取得最近一次 pipeline 的角度結果。
   * @param superResolutionNativeHandle initWithByteBufferFromJNI 回傳的 native 物件位址
   * @return {絕對角, 相對角}（度）；無有效結果時 {-1, -1}；失敗 null
   */
  private native double[] PW_DOPPLER_ANGLE(long superResolutionNativeHandle);

  /**
   * 取得中心線與取樣線交點（或備援中位點）的座標。
   * @param superResolutionNativeHandle native 物件位址
   * @return {x, y}（原圖像素座標）；無有效結果時 {-1, -1}；失敗 null
   */
  private native int[] PW_Center_Point(long superResolutionNativeHandle);

  /**
   * 取得 Range Gate 上下位置的 y 座標（中心點往血管上下邊界各 kRangeGateRatio 處）。
   * @param superResolutionNativeHandle native 物件位址
   * @return {top_y, bottom_y}；無有效結果時 {-1, -1}；失敗 null
   */
  private native int[] PW_RANGE_GATE(long superResolutionNativeHandle);

  /**
   * 取得沿取樣線量到的血管寬度（上下邊界交點的距離）。
   * @param superResolutionNativeHandle native 物件位址
   * @return 寬度（像素）；無有效結果時 -1
   */
  private native double PW_VESSEL_DIAMETER(long superResolutionNativeHandle);

  /**
   * 偵測影像上的綠色都卜勒取樣線。
   * @param imageData ARGB 像素陣列
   * @param width     影像寬
   * @param height    影像高
   * @return {角度（度，垂直為 0、右偏為正）, 頂端 x}；找不到綠線時 {9999, -1}
   */
  private native double[] detectLineFromJNI(int[] imageData, int width, int height);

  /**
   * 去除影像上的紅 / 綠色 UI 疊圖（取樣線、血流色標等），以 inpaint 填補。
   * @param imageData ARGB 像素陣列
   * @param width     影像寬
   * @param height    影像高
   * @return 處理後的 ARGB 像素陣列（同尺寸）
   */
  private native int[] removeGreenRedFromJNI(int[] imageData, int width, int height);

  /**
   * 建立畫面：依螢幕寬度設定結果圖尺寸、載入三張測試影格、綁定點選與執行按鈕。
   * 按下 Upsample 後依序呼叫 detectLineFromJNI → removeGreenRedFromJNI → doSuperResolution，
   * 再用 PW_* 取得數值並顯示於 log_view。
   * @param savedInstanceState Activity 狀態（未使用）
   */
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    ImageView superResolutionImageView = findViewById(R.id.super_resolution_image);

    // 根據屏幕寬度計算顯示尺寸
    DisplayMetrics displayMetrics = new DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
    int screenWidth = displayMetrics.widthPixels;

    // 設置 ImageView 的 layout 參數
    ViewGroup.LayoutParams params = superResolutionImageView.getLayoutParams();
    params.width = screenWidth - 32; // 左右留一些邊距
    params.height = (int)((float)SR_IMAGE_HEIGHT / SR_IMAGE_WIDTH * params.width);
    superResolutionImageView.setLayoutParams(params);
    superResolutionImageView.setScaleType(ImageView.ScaleType.FIT_CENTER);


    final Button superResolutionButton = findViewById(R.id.upsample_button);
    lowResImageView1 = findViewById(R.id.low_resolution_image_1);
    lowResImageView2 = findViewById(R.id.low_resolution_image_2);
    lowResImageView3 = findViewById(R.id.low_resolution_image_3);
    selectedImageTextView = findViewById(R.id.chosen_image_tv);

    ImageView[] lowResImageViews = {lowResImageView1, lowResImageView2, lowResImageView3};

    AssetManager assetManager = getAssets();
    try {
      InputStream inputStream1 = assetManager.open(LR_IMG_1);
      Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream1);
      lowResImageView1.setImageBitmap(bitmap1);

      InputStream inputStream2 = assetManager.open(LR_IMG_2);
      Bitmap bitmap2 = BitmapFactory.decodeStream(inputStream2);
      lowResImageView2.setImageBitmap(bitmap2);

      InputStream inputStream3 = assetManager.open(LR_IMG_3);
      Bitmap bitmap3 = BitmapFactory.decodeStream(inputStream3);
      lowResImageView3.setImageBitmap(bitmap3);
    } catch (IOException e) {
      Log.e(TAG, "Failed to open an low resolution image");
    }

    for (ImageView iv : lowResImageViews) {
      setLRImageViewListener(iv);
    }

    superResolutionButton.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View view) {
            if (selectedLRBitmap == null) {
              Toast.makeText(
                      getApplicationContext(),
                      "Please choose one low resolution image",
                      Toast.LENGTH_LONG)
                  .show();
              return;
            }

            if (superResolutionNativeHandle == 0) {
                superResolutionNativeHandle = initTFLiteInterpreter();
            }
            if (superResolutionNativeHandle == 0) {
              showToast("TFLite interpreter failed to create!");
              return;
            }

            int[] lowResRGB = new int[LR_IMAGE_HEIGHT * LR_IMAGE_WIDTH];
            selectedLRBitmap.getPixels(
                lowResRGB, 0, LR_IMAGE_WIDTH, 0, 0, LR_IMAGE_WIDTH, LR_IMAGE_HEIGHT);

            StringBuilder logText = new StringBuilder();

            // 1. 偵測線段（模擬從外部獲取）
            double[] lineInfo = detectLineFromJNI(lowResRGB, LR_IMAGE_WIDTH, LR_IMAGE_HEIGHT);
            double lineAngle = lineInfo[0];
            int topX = (int) lineInfo[1];
            logText.append(String.format("line angle: %.2f\n", lineAngle));
            logText.append(String.format("topX: %d\n", topX));
            logText.append(String.format("\n"));

            // 2. 去除綠紅線（模擬從外部獲取處理後的圖像）
            int[] cleanedImage = removeGreenRedFromJNI(lowResRGB, LR_IMAGE_WIDTH, LR_IMAGE_HEIGHT);

            final long startTime = SystemClock.uptimeMillis();
            int[] superResRGB = doSuperResolution(cleanedImage, lineAngle, topX);
            final long processingTimeMs = SystemClock.uptimeMillis() - startTime;
            if (superResRGB == null) {
              showToast("Super resolution failed!");
              return;
            }

            // 獲取角度
            double[] angles = PW_DOPPLER_ANGLE(superResolutionNativeHandle);
            if (angles != null && angles.length >= 2 && angles[0] >= 0) {
              double absoluteAngle = angles[0];
              double relativeAngle = angles[1];
              Log.d(TAG, "Absolute angle: " + absoluteAngle);
              Log.d(TAG, "Relative angle: " + relativeAngle);
            }

            // 獲取中心點
            int[] center = PW_Center_Point(superResolutionNativeHandle);
            if (center != null && center.length >= 2 && center[0] >= 0) {
              int centerX = center[0];
              int centerY = center[1];
              Log.d(TAG, "Center point: (" + centerX + ", " + centerY + ")");
            }

            // 獲取 Range Gate
            int[] rangeGate = PW_RANGE_GATE(superResolutionNativeHandle);
            if (rangeGate != null && rangeGate.length >= 2 && rangeGate[0] >= 0) {
              int topY = rangeGate[0];
              int bottomY = rangeGate[1];
              Log.d(TAG, "Range gate Y: top=" + topY + ", bottom=" + bottomY);

              int pixel_vessel_width = Math.abs(bottomY - topY);
              Log.d(TAG, "Vessel Width (pixels): " + pixel_vessel_width);
            }

            // 獲取 Vessel Diameter (綠線處血管管徑寬度)
            double vesselDiameter = PW_VESSEL_DIAMETER(superResolutionNativeHandle);

            final LinearLayout resultLayout = findViewById(R.id.result_layout);
            final ImageView superResolutionImageView = findViewById(R.id.super_resolution_image);
            final ImageView nativelyScaledImageView = findViewById(R.id.natively_scaled_image);
            final TextView superResolutionTextView = findViewById(R.id.super_resolution_tv);
            final TextView nativelyScaledImageTextView =
                findViewById(R.id.natively_scaled_image_tv);
            final TextView logTextView = findViewById(R.id.log_view);

            resultLayout.setVisibility(View.VISIBLE);

            // Force refreshing the ImageView
            superResolutionImageView.setImageDrawable(null);
            Bitmap srImgBitmap =
                    Bitmap.createBitmap(
                            superResRGB, SR_IMAGE_WIDTH, SR_IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
            superResolutionImageView.setImageBitmap(srImgBitmap);
            nativelyScaledImageView.setImageBitmap(selectedLRBitmap);
            resultLayout.setVisibility(View.VISIBLE);

            logText.append("Inference time: ").append(processingTimeMs).append("ms\n");

            if (angles != null && angles.length >= 2) {
              logText.append(String.format("  Absolute angle : %.2f°\n", angles[0]));
              logText.append(String.format("  Relative angle : %.2f°\n", angles[1]));
            } else {
              logText.append("✗ Get angle failed \n");
            }

            if (center != null && center.length >= 2 && center[0] >= 0) {
              logText.append(String.format("  center point: (%d, %d)\n", center[0], center[1]));
            }

            if (rangeGate != null && rangeGate.length >= 2 && rangeGate[0] >= 0) {
              logText.append(String.format("  Range Gate Y: up = %d,  down = %d\n", rangeGate[0], rangeGate[1]));
            }

            if (vesselDiameter >= 0) {
              logText.append(String.format("  Vessel Width: %.2f pixels\n", vesselDiameter));
            } else {
              logText.append("  Vessel Width: failed\n");
            }

            logTextView.setText(logText.toString());
          }
        });


      Toast.makeText(this, "OpenCV & 平板連線成功", Toast.LENGTH_SHORT).show();
  }

  /**
   * Activity 結束時釋放 native 物件。
   */
  @Override
  public void onDestroy() {
    super.onDestroy();
    deinit();
  }

  /**
   * 為測試影格的 ImageView 加上觸控監聽：點選後把該 Bitmap 設為待處理影像並更新提示文字。
   * @param iv 三張測試影格之一的 ImageView
   */
  private void setLRImageViewListener(ImageView iv) {
    iv.setOnTouchListener(
        new View.OnTouchListener() {
          @Override
          public boolean onTouch(View v, MotionEvent event) {
            if (v.equals(lowResImageView1)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView1.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 1 ("
                      + getResources().getString(R.string.low_resolution_1)
                      + ")");
            } else if (v.equals(lowResImageView2)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView2.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 2 ("
                      + getResources().getString(R.string.low_resolution_2)
                      + ")");
            } else if (v.equals(lowResImageView3)) {
              selectedLRBitmap = ((BitmapDrawable) lowResImageView3.getDrawable()).getBitmap();
              selectedImageTextView.setText(
                  "You are using low resolution image: 3 ("
                      + getResources().getString(R.string.low_resolution_3)
                      + ")");
            }
            return false;
          }
        });
  }

  /**
   * 執行完整 native pipeline（前處理 → 模型推論 → 後處理 → 視覺化）。
   * @param lowResRGB LR_IMAGE_WIDTH×LR_IMAGE_HEIGHT 的 ARGB 像素陣列（建議先經 removeGreenRedFromJNI）
   * @param lineAngle detectLineFromJNI 得到的取樣線角度
   * @param topX      detectLineFromJNI 得到的取樣線頂端 x
   * @return 同尺寸的 ARGB 視覺化影像；失敗時 null
   */
  @WorkerThread
  public synchronized int[] doSuperResolution(int[] lowResRGB, double lineAngle, int topX) {
    return superResolutionFromJNI(superResolutionNativeHandle, lowResRGB, lineAngle, topX);
  }

  /**
   * 把 assets 內的模型檔（MODEL_NAME）以唯讀方式 memory-map 進來。
   * @return 模型的 MappedByteBuffer
   * @throws IOException 模型檔開啟或映射失敗
   */
  private MappedByteBuffer loadModelFile() throws IOException {
    try (AssetFileDescriptor fileDescriptor =
            AssetsUtil.getAssetFileDescriptorOrCached(getApplicationContext(), MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /**
   * 顯示一則長時間 Toast。
   * @param str 要顯示的文字
   */
  private void showToast(String str) {
    Toast.makeText(getApplicationContext(), str, Toast.LENGTH_LONG).show();
  }

  /**
   * 載入模型並在 native 端建立 TFLite interpreter。
   * @return native 物件位址；建立失敗時 0
   */
  private long initTFLiteInterpreter() {
    try {
      model = loadModelFile();
      Log.d(TAG, "Model loaded, capacity: " + (model != null ? model.capacity() : "null"));
    } catch (IOException e) {
      Log.e(TAG, "Fail to load model", e);
      return 0;   // 模型缺檔：不呼叫 JNI，讓呼叫端顯示 "TFLite interpreter failed to create!"
    }
    return initWithByteBufferFromJNI(model);
  }

  /**
   * 釋放 native 端的 SuperResolution 物件。
   */
  private void deinit() {
    deinitFromJNI(superResolutionNativeHandle);
  }

  /**
   * JNI：執行完整 pipeline 並回傳視覺化影像（見 doSuperResolution）。
   * @param superResolutionNativeHandle native 物件位址
   * @param lowResRGB  ARGB 像素陣列
   * @param line_angle 取樣線角度
   * @param top_x      取樣線頂端 x
   * @return ARGB 視覺化影像；失敗時 null
   */
  private native int[] superResolutionFromJNI(long superResolutionNativeHandle, int[] lowResRGB, double line_angle, int top_x);

  /**
   * JNI：由 direct ByteBuffer 中的 .tflite 建立 native SuperResolution 物件。
   * @param modelBuffer 映射模型檔的 MappedByteBuffer
   * @return native 物件位址；失敗時 0
   */
  private native long initWithByteBufferFromJNI(MappedByteBuffer modelBuffer);

  /**
   * JNI：釋放 initWithByteBufferFromJNI 建立的 native 物件。
   * @param superResolutionNativeHandle native 物件位址
   */
  private native void deinitFromJNI(long superResolutionNativeHandle);
}