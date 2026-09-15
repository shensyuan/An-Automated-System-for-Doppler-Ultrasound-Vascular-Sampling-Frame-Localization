# 自動化都普勒血管取樣框定位系統 — Android 端
An Automated System for Doppler Ultrasound Vascular Sampling Frame Localization (Android deployment)

Demo video：<https://drive.google.com/file/d/1T_JKMZ3sjOHdsewaSxW32veMggLWTDnQ/view>

## 1. 系統總覽

本專案是系統的**端側部署**部分：在 Android 裝置上載入訓練好的血管分割 `.tflite` 模型，對一張超音波影格（900×700）執行

> 取樣線偵測 → 去除紅綠 UI 疊圖 → 前處理 → 模型推論 → 後處理（中心線 / 切線 / Range Gate）→ 視覺化

輸出疊加了取樣線、血流切線與 Range Gate 標記的影像，以及角度、中心點、Range Gate 位置、血管寬度等量測數值。

專案由 TensorFlow Lite 官方 Super Resolution 範例改寫而來，因此 class / lib / 模型檔名（`SuperResolution`、`ESRGAN.tflite`）沿用範例名稱，**實際內容是血管分割 pipeline 相關**。

## 2. 處理流程

進入點是 [`MainActivity.java`](app/src/main/java/org/tensorflow/lite/examples/superresolution/MainActivity.java)：使用者點選一張測試影格、按下 `Upsample` 後，依序透過 JNI（[`SuperResolution_jni.cpp`](app/src/main/cc/SuperResolution_jni.cpp)）呼叫 native 端的 [`SuperResolution`](app/src/main/cc/SuperResolution.cpp) 類別。

| 步驟 | 執行端 | 函式 | 說明 |
|---|---|---|---|
| 1. 取樣線偵測 | JNI | `detectLineFromJNI` | 以 BGR 範圍取綠色像素，最高 / 最低點連線求取樣線角度與頂端 x |
| 2. 去除紅綠疊圖 | JNI | `removeGreenRedFromJNI` | HSV 遮罩（綠 H 40~80、紅 H 0~10 / 170~180）膨脹後以 Telea inpaint 填補 |
| 3. 前處理 | `DoSuperResolution` | `get_line_point` → `resize_with_padding` → 轉灰階 → `apply_clahe` | 由角度算出取樣線端點；等比縮放補邊到 224×224；CLAHE 對比強化 |
| 4. 模型推論 | `doseg` | — | 正規化 (x/127.5−1) 餵進 TFLite，輸出 224×224 mask，`val×255 > 128` 為前景 |
| 5. 後處理 | `postprocess` | `process_single_centerline`、`get_tangent_direction`、`find_RangeGate`、`calculate_angle_between_vectors` | 見下 |
| 6. 視覺化 | `visualizePostProcess` | `draw_tangent`、`draw_perpendicular_line` | 綠色取樣線（畫到 Range Gate 處）、紅色切線、紅色 Range Gate 短線 |
| 7. 取數值 | JNI | `PW_DOPPLER_ANGLE`、`PW_Center_Point`、`PW_RANGE_GATE`、`PW_VESSEL_DIAMETER` | 讀取 `last_result_`，見第 5 節 |

### 後處理細節（`postprocess`）

1. **中心線擷取** `process_single_centerline`：mask 二值化 → 8 連通元件（<20 px 略過）→ 距離轉換 → Zhang-Suen 細線化（`thinningZhangSuen`）→ 每個 x 欄取距離值最大的骨架點 → `is_valid_centerline` 篩選 → 座標縮放回原圖 → Catmull-Rom 樣條（`generateSplinePoints`）內插成平滑中心線。
2. **決定 center 與線束**
   - 外部取樣線與中心線**有交點**：以第一個交點為 `center`，線束沿用外部取樣線。
   - **無交點（備援）**：取中心線中位點為 `center`，以 `kFallbackBeamAngleDeg` 通過 `center` 重畫一條線束（`get_boundary_intersection_direct`）。
3. **切線方向** `get_tangent_direction`：取 `center` 周圍 ±15 px 的骨架點做 PCA（SVD），第一主成分即血流方向。
4. **Range Gate 邊界** `find_RangeGate`：從 `center` 沿線束往上下逐像素走，離開血管 mask 時回退 2 px 即為 `intersection_top` / `intersection_bottom`。
5. **角度** `calculate_angle_between_vectors`：`angle_abs`＝切線 vs 垂直軸、`angle_relative`＝切線 vs 線束。

## 3. 關鍵常數

| 常數 | 位置 | 值 | 意義 |
|---|---|---|---|
| `inputHeight` / `inputWidth` | [`SuperResolution.h`](app/src/main/cc/SuperResolution.h) | 700 / 900 | 輸入影格解析度（Java 端 `LR_IMAGE_*` 需一致） |
| `kRangeGateRatio` | `SuperResolution.h` | 2/3 | Range Gate 位置：從 `center` 往血管上下邊界前進的比例 |
| `kFallbackBeamAngleDeg` | `SuperResolution.h` | 75 | 備援線束角度（相對 +x 軸、順時針為正；75° 即偏離垂直 15°），找不到取樣線與中心線交點時使用 |
| `kThreadNum` | [`SuperResolution.cpp`](app/src/main/cc/SuperResolution.cpp) | 4 | TFLite interpreter 執行緒數 |
| 模型輸入 / 輸出 | `doseg` | 224×224 | 由 tensor 維度讀取，通道數 1 或 3 皆可 |
| mask 閾值 | `doseg` | 128 | `val×255 > 128` 視為前景 |
| 切線 PCA 視窗 | `postprocess` | 15 px | `get_tangent_direction` 的 `window_size` |
| 中心線篩選 | `is_valid_centerline` | ≥15 點、寬 ≥20 px、y 變異 ≤1.5 | 過濾過短 / 過窄 / 抖動的片段 |

## 4. 函式對應

| 階段 | 函式（`SuperResolution.cpp`） |
|---|---|
| 前處理 | `apply_clahe`、`resize_with_padding`、`get_line_point` |
| 尺寸轉換 | `resize_img`、`crop_img` |
| 中心線 | `thinningZhangSuen`、`is_valid_centerline`、`catmullRomPoint`、`generateSplinePoints`、`process_single_centerline` |
| 線束 / Range Gate | `find_RangeGate`、`get_boundary_intersection_direct`、`get_tangent_direction`、`calculate_angle_between_vectors` |
| 繪圖 | `draw_tangent`、`draw_perpendicular_line`、`visualizePostProcess` |
| 模型 | 建構子（載入模型、建立 interpreter）、`IsInterpreterCreated`、`doseg` |
| 整合入口 | `DoSuperResolution`（步驟 1~9）、`postprocess` |

每個函式的功能、輸入、輸出都寫在定義上方的 Doxygen 註解裡。

## 5. JNI API（`MainActivity` 的 `native` 方法）

| Java 宣告 | 回傳 | 失敗值 |
|---|---|---|
| `long initWithByteBufferFromJNI(MappedByteBuffer)` | native 物件位址 | `0` |
| `void deinitFromJNI(long)` | — | — |
| `double[] detectLineFromJNI(int[] argb, int w, int h)` | `{角度(度), 頂端 x}` | `{9999, -1}` |
| `int[] removeGreenRedFromJNI(int[] argb, int w, int h)` | 去疊圖後的 ARGB 陣列 | `null` |
| `int[] superResolutionFromJNI(long, int[] argb, double angle, int topX)` | 視覺化 ARGB 陣列 | `null` |
| `double[] PW_DOPPLER_ANGLE(long)` | `{angle_abs, angle_relative}` | `{-1, -1}` |
| `int[] PW_Center_Point(long)` | `{x, y}` | `{-1, -1}` |
| `int[] PW_RANGE_GATE(long)` | `{top_y, bottom_y}`（`kRangeGateRatio` 處） | `{-1, -1}` |
| `double PW_VESSEL_DIAMETER(long)` | 上下邊界交點距離（px） | `-1.0` |

`PW_*` 讀的是最近一次 `superResolutionFromJNI` 存在 `last_result_` 的結果，必須在其之後呼叫。

## 6. 檔案地圖

```
android/
├── README.md
├── build.gradle / settings.gradle / gradle.properties / gradlew(.bat) / gradle/
│                              ← Gradle 8.11.1 專案骨架（AGP 8.9.1）
├── app/
│   ├── build.gradle           ← compileSdk 36、NDK 27.0.12077973、CMake 3.22.1；-DOpenCV_DIR 指到 opencv/native
│   ├── download.gradle        ← fetchTFLiteLibs 等下載 task（見第 7 節注意事項）
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/            ← ESRGAN.tflite（血管分割模型）+ 測試影格（data1_clip_*.png、data6_*.png 等）
│       ├── cc/                ← ★ native pipeline
│       │   ├── CMakeLists.txt
│       │   ├── SuperResolution.h / .cpp    ← pipeline 邏輯與可調常數
│       │   └── SuperResolution_jni.cpp     ← JNI 橋接
│       ├── java/.../superresolution/
│       │   ├── MainActivity.java   ← App 進入點、模型載入、UI 綁定
│       │   └── AssetsUtil.java     ← asset 讀取（壓縮 asset 會先複製到 cache）
│       ├── jniLibs/<ABI>/libopencv_java4.so
│       └── res/               ← layout（activity_main、bottom_sheet）、strings、icon
├── libraries/
│   └── tensorflowlite/        ← TFLite C API headers + .so（由 fetchTFLiteLibs 解出） 
└── opencv/                    ← OpenCV Android SDK（native/jni/include 供 cc/ 使用）
```

## 7. 建置與執行

1. **環境**：Android Studio、Android SDK（compileSdk 36）、NDK `27.0.12077973`、CMake `3.22.1`。
   `local.properties` 的 `sdk.dir` 必須指向本機 SDK；命令列建置可改用環境變數 `ANDROID_HOME`。
2. **TFLite 函式庫**：若 `libraries/tensorflowlite*/` 不存在，在 `android/` 執行
   ```
   gradlew.bat fetchTFLiteLibs        # Windows
   ./gradlew fetchTFLiteLibs          # macOS / Linux
   ```
3. 用 Android Studio 開啟 `android/`，等待 Gradle sync 與 CMake 設定完成後執行 `app`。
4. App 內點選三張測試影格之一（`LR_IMG_1~3`，對應 `assets/data1_clip_141s_5.83s.png`、`data1_clip_560s_6.81s.png`、`data6_0.95s.png`），按 `Upsample`；結果圖與量測數值會顯示在下方。

命令列只驗證編譯：
```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:externalNativeBuildDebug :app:compileDebugJavaWithJavac
```

### 注意事項

- 模型檔 `assets/ESRGAN.tflite`（血管分割模型）已進版控，必須隨 repo 一起存在；Gradle **不會**自動下載任何模型（原 TFLite 範例的 `downloadESRGANModelFile` task 已移除，避免缺檔時抓到真正的 ESRGAN 超解析模型）。缺檔時 App 會在按下 `Upsample` 後顯示 `TFLite interpreter failed to create!`。
- 要換輸入解析度時，`SuperResolution.h` 的 `inputHeight/inputWidth` 與 `MainActivity.java` 的 `LR_IMAGE_*` / `SR_IMAGE_*` 需同步修改。
- 要調整備援線束角度或 Range Gate 位置，改 `SuperResolution.h` 的 `kFallbackBeamAngleDeg` / `kRangeGateRatio` 即可，不需動其他程式碼。
