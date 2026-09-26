/* HyperIDs — ONNX Runtime C API inference example (deployment reference).
 *
 * Build (paths for the onnxruntime wheel):
 *   gcc -O2 c_infer_example.c -I<site-packages>/onnxruntime/capi \
 *       -L<site-packages>/onnxruntime/capi -lonnxruntime -o c_infer_example
 *
 * Usage:
 *   ./c_infer_example <model.onnx> <input_ids.bin> <attention_mask.bin>
 * where the .bin files are int64 arrays of shape [1, SEQ_LEN] (SEQ_LEN=384),
 * produced by export/prepare_c_input.py (edge: SEQ_LEN=320, N_LOGITS=25).
 */
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static const OrtApi* g_ort = NULL;

#define SEQ_LEN 320
#define N_LOGITS 31

static void check(OrtStatus* st) {
    if (st) {
        const char* msg = g_ort->GetErrorMessage(st);
        fprintf(stderr, "ORT error: %s\n", msg);
        g_ort->ReleaseStatus(st);
        exit(1);
    }
}

static int64_t* read_bin(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int64_t* buf = (int64_t*)malloc(n * sizeof(int64_t));
    if (fread(buf, sizeof(int64_t), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.onnx> <input_ids.bin> <attention_mask.bin>\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    int64_t* input_ids = read_bin(argv[2], SEQ_LEN);
    int64_t* attn_mask = read_bin(argv[3], SEQ_LEN);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    OrtSessionOptions* opts = NULL;
    OrtSession* session = NULL;
    OrtMemoryInfo* mem_info = NULL;

    check(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hyperids", &env));
    check(g_ort->CreateSessionOptions(&opts));
    g_ort->SetIntraOpNumThreads(opts, 1);
    /* keep peak RSS low for edge deployment (<100MB) */
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL);
    g_ort->DisableMemPattern(opts);
    g_ort->DisableCpuMemArena(opts);
    check(g_ort->CreateSession(env, model_path, opts, &session));
    check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    const int64_t shape[2] = {1, SEQ_LEN};
    OrtValue* input_t = NULL;
    OrtValue* mask_t = NULL;
    check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, input_ids, SEQ_LEN * sizeof(int64_t), shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_t));
    check(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, attn_mask, SEQ_LEN * sizeof(int64_t), shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &mask_t));

    const char* input_names[2] = {"input_ids", "attention_mask"};
    const char* output_names[1] = {"logits"};
    OrtValue* outputs[1] = {NULL};

    check(g_ort->Run(session, NULL, input_names, (const OrtValue* const[]){input_t, mask_t},
                 2, output_names, 1, outputs));

    float* logits = NULL;
    check(g_ort->GetTensorMutableData(outputs[0], (void**)&logits));

    /* logits are [1, N_LOGITS]; the first k are the chunk's label scores */
    printf("logits (first 10):\n");
    for (int i = 0; i < 10 && i < N_LOGITS; i++) printf("  [%d] %.4f\n", i, logits[i]);

    g_ort->ReleaseValue(outputs[0]);
    g_ort->ReleaseValue(input_t);
    g_ort->ReleaseValue(mask_t);
    g_ort->ReleaseMemoryInfo(mem_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);
    free(input_ids);
    free(attn_mask);
    printf("OK\n");
    return 0;
}
