#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <string>
#include <filesystem>

#include "pre.h"
#include "post.h"

namespace fs = std::filesystem;

int main() {
    // ====== Paths ======
    std::string input_image_dir = "input_data/input_image";
    std::string test_mask_dir   = "test_data/test_masks";
    std::string output_dir      = "./result";

    std::map<std::string, FrameData> frames;
    int image_h = 0, image_w = 0;

    // ====== Scan original images (with subdirectories) ======
    for (const auto& entry : fs::recursive_directory_iterator(input_image_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

        std::string fname     = entry.path().filename().string();
        std::string stem      = entry.path().stem().string();  // e.g. "data6_0.95s"
        fs::path mask_path    = fs::path(test_mask_dir) / (stem + "_label.png");

        // ====== Read original image (color, needed for line detection) ======
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Cannot read image: " << entry.path() << std::endl;
            continue;
        }

        // ====== Read corresponding mask (grayscale, replaces model output) ======
        cv::Mat mask = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            std::cout << "Cannot find mask for " << fname << " at " << mask_path << std::endl;
            continue;
        }
        // Resize mask to 224x224 (expected by process_single_centerline)
        cv::resize(mask, mask, cv::Size(224, 224));

        image_h = img.rows;
        image_w = img.cols;

        // ====== Detect Doppler beam line (pre.cpp) ======
        auto [line_angle, top_x] = detected_line(img);

        std::vector<cv::Point> line(2);
        if (top_x < 0) {
            std::cout << fname << " no line detected, using default" << std::endl;
            line[0] = cv::Point(0, 0);
            line[1] = cv::Point(0, 0);
        } else {
            line[0] = cv::Point(top_x, 0);
            line[1] = get_line_point(line[0].x, line[0].y, line_angle, image_h);
            std::cout << fname << " angle: " << line_angle << " top_x: " << top_x << std::endl;
        }

        cv::Mat result = remove_green_red(img);
        cv::Mat padded = resize_with_padding(result, 224);
        cv::Mat gray;
        cv::cvtColor(padded, gray, cv::COLOR_BGR2GRAY);
        cv::Mat enhanced = apply_clahe(gray);

        // ====== Build frame data ======
        FrameData frame;
        frame.img       = enhanced;
        frame.raw_mask  = mask;
        frame.line      = line;

        frames[fname] = frame;
    }

    if (frames.empty()) {
        std::cout << "No valid frame pairs found." << std::endl;
        return -1;
    }

    std::cout << "Total " << frames.size() << " frames to process." << std::endl;

    // ====== Run post-processing ======
    post_process(frames, image_h, image_w, output_dir);

    std::cout << "Done." << std::endl;
    return 0;
}
