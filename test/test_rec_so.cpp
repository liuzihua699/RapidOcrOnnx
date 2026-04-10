#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "OcrLiteCApi.h"

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <rec_model> <keys_file> <image> <gpu_index> <loop_count>\n", argv[0]);
        return 1;
    }

    const char *recModelPath = argv[1];
    const char *keysPath     = argv[2];
    const char *imgPath      = argv[3];
    int gpuIndex  = atoi(argv[4]);
    int loopCount = atoi(argv[5]);

    cv::Mat src = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        fprintf(stderr, "Failed to read image: %s\n", imgPath);
        return 1;
    }

    printf("=====Init rec model (GPU %d)=====\n", gpuIndex);
    OCR_HANDLE handle = OcrRecInit(recModelPath, keysPath, 4, gpuIndex);
    if (!handle) {
        fprintf(stderr, "OcrRecInit failed\n");
        return 1;
    }

    printf("=====Warmup 2 cycles=====\n");
    for (int i = 0; i < 2; ++i) {
        REC_RESULT result;
        memset(&result, 0, sizeof(result));
        OCR_BOOL ok = OcrRecDetect(handle, src.data, src.cols, src.rows, src.channels(), &result);
        if (ok) {
            printf("Warmup time(%f) text=%s\n", result.crnnTime, (const char *)result.text);
            OcrRecFreeResult(&result);
        } else {
            printf("Warmup failed\n");
        }
    }

    printf("=====Start Test Loop=====\n");
    double allRecTime = 0.0;

    for (int i = 0; i < loopCount; ++i) {
        printf("=====Cycle:%d Take Time(ms)=====\n", i + 1);
        REC_RESULT result;
        memset(&result, 0, sizeof(result));
        OCR_BOOL ok = OcrRecDetect(handle, src.data, src.cols, src.rows, src.channels(), &result);
        if (ok) {
            printf("rec=%f text=%s\n", result.crnnTime, (const char *)result.text);
            allRecTime += result.crnnTime;
            OcrRecFreeResult(&result);
        } else {
            printf("rec failed at cycle %d\n", i + 1);
        }
    }

    double avg = allRecTime / loopCount;
    printf("=====Result:Average Time(ms)=====\n");
    printf("rec=%f\n", avg);
    printf("loops=%d\n", loopCount);
    printf("throughput=%.1f img/s\n", 1000.0 / avg);

    OcrRecDestroy(handle);
    return 0;
}
