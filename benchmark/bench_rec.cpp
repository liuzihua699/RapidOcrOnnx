#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "OcrStruct.h"
#include "OcrUtils.h"
#include "CrnnNet.h"



struct ImageEntry {
    std::string path;
    cv::Mat mat;
};

std::vector<ImageEntry> loadImagesFromDir(const std::string &imagesDir) {
    std::vector<ImageEntry> images;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};
    for (const auto &entry : std::filesystem::directory_iterator(imagesDir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) continue;
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            fprintf(stderr, "Warning: failed to load image: %s\n", entry.path().c_str());
            continue;
        }
        images.push_back({entry.path().string(), std::move(img)});
    }
    printf("Loaded %zu images from %s\n", images.size(), imagesDir.c_str());
    return images;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <rec_model> <keys_file> <image> <gpu_index> <loop_count>\n", argv[0]);
        return 1;
    }



    // 预载图片
    std::string imagesDir = "/root/rapid_ocr/images";
    std::vector<ImageEntry> images = loadImagesFromDir(imagesDir);
    if (images.empty()) {
        fprintf(stderr, "No images found in: %s\n", imagesDir.c_str());
        return -1;
    }

    // 随机数生成器
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, images.size() - 1);


    std::string recModelPath = argv[1];
    std::string keysPath = argv[2];
    std::string imgPath = argv[3];
    int gpuIndex = atoi(argv[4]);
    int loopCount = atoi(argv[5]);

    cv::Mat src = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        fprintf(stderr, "Failed to read image: %s\n", imgPath.c_str());
        return 1;
    }

    CrnnNet crnnNet;
    crnnNet.setNumThread(4);
    crnnNet.setGpuIndex(gpuIndex);
    crnnNet.initModel(recModelPath, keysPath);

    std::vector<cv::Mat> partImgs;
    partImgs.push_back(src);

    printf("=====Warmup 2 cycles=====\n");
    for (int i = 0; i < 2; ++i) {
        std::vector<TextLine> results = crnnNet.getTextLines(partImgs, "/tmp/", "warmup");
        printf("Warmup time(%f)\n", results[0].time);
    }


    printf("=====Start Test Loop=====\n");
    double allRecTime = 0.0;

    for (int i = 0; i < loopCount; ++i) {
        printf("=====Cycle:%d Take Time(ms)=====\n", i + 1);
    
        // size_t idx = dist(rng);
        // partImgs[0] = images[idx].mat;
        // printf("Processing image: %s\n", images[idx].path.c_str());

        std::vector<TextLine> results = crnnNet.getTextLines(partImgs, "/tmp/", "bench");
        double recTime = results[0].time;
        printf("rec=%f text=%s\n", recTime, results[0].text.c_str());
        allRecTime += recTime;
    }

    double avg = allRecTime / loopCount;
    printf("=====Result:Average Time(ms)=====\n");
    printf("rec=%f\n", avg);
    printf("loops=%d\n", loopCount);
    printf("throughput=%.1f img/s\n", 1000.0 / avg);

    return 0;
}
