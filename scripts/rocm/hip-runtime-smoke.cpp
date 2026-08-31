// A real kernel-launch probe for Windows ROCm/TheRock packages.
// hipGetDeviceCount() and hipInfo can succeed even when the installed display
// driver rejects every code object produced by the SDK.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdio>
#include <cstring>

__global__ void beellama_hip_smoke_add_one(int * value) {
    if (threadIdx.x == 0) {
        *value += 1;
    }
}

static int fail(const char * operation, hipError_t status) {
    std::fprintf(stderr, "HIP runtime smoke test: %s failed: %s\n",
            operation, hipGetErrorString(status));
    return 1;
}

static int fail_blas(const char * operation, hipblasStatus_t status) {
    std::fprintf(stderr, "HIP runtime smoke test: %s failed (hipBLAS status %d)\n",
            operation, int(status));
    return 1;
}

int main() {
    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        return fail("hipGetDeviceCount", status);
    }

    int device = -1;
    hipDeviceProp_t selected {};
    for (int candidate = 0; candidate < count; ++candidate) {
        hipDeviceProp_t properties {};
        status = hipGetDeviceProperties(&properties, candidate);
        if (status != hipSuccess) {
            return fail("hipGetDeviceProperties", status);
        }
        if (std::strncmp(properties.gcnArchName, "gfx1100", 7) == 0) {
            device = candidate;
            selected = properties;
            break;
        }
    }
    if (device < 0) {
        std::fprintf(stderr, "HIP runtime smoke test: no gfx1100 device was found\n");
        return 2;
    }
    status = hipSetDevice(device);
    if (status != hipSuccess) {
        return fail("hipSetDevice", status);
    }

    int host = 41;
    int * allocation = nullptr;
    status = hipMalloc(&allocation, sizeof(host));
    if (status != hipSuccess) {
        return fail("hipMalloc", status);
    }
    status = hipMemcpy(allocation, &host, sizeof(host), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        hipFree(allocation);
        return fail("host-to-device copy", status);
    }

    beellama_hip_smoke_add_one<<<1, 1>>>(allocation);
    status = hipGetLastError();
    if (status == hipSuccess) {
        status = hipDeviceSynchronize();
    }
    if (status == hipSuccess) {
        status = hipMemcpy(&host, allocation, sizeof(host), hipMemcpyDeviceToHost);
    }
    hipFree(allocation);
    if (status != hipSuccess) {
        return fail("gfx1100 kernel launch", status);
    }
    if (host != 42) {
        std::fprintf(stderr, "HIP runtime smoke test: invalid kernel result %d\n", host);
        return 3;
    }

    // Force rocBLAS to resolve and launch a Tensile kernel as well. A package
    // can pass ordinary HIP initialization while still missing the
    // architecture-specific rocblas/library payload needed by batched MTP
    // catch-up.
    hipblasHandle_t blas = nullptr;
    hipblasStatus_t blas_status = hipblasCreate(&blas);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        return fail_blas("hipblasCreate", blas_status);
    }
    float a = 6.0f;
    float b = 7.0f;
    float c = 0.0f;
    float * d_a = nullptr;
    float * d_b = nullptr;
    float * d_c = nullptr;
    status = hipMalloc(&d_a, sizeof(float));
    if (status == hipSuccess) status = hipMalloc(&d_b, sizeof(float));
    if (status == hipSuccess) status = hipMalloc(&d_c, sizeof(float));
    if (status == hipSuccess) status = hipMemcpy(d_a, &a, sizeof(float), hipMemcpyHostToDevice);
    if (status == hipSuccess) status = hipMemcpy(d_b, &b, sizeof(float), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
        hipblasDestroy(blas);
        return fail("hipBLAS smoke allocation", status);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    blas_status = hipblasSgemm(
            blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
            1, 1, 1, &alpha, d_a, 1, d_b, 1, &beta, d_c, 1);
    if (blas_status == HIPBLAS_STATUS_SUCCESS) {
        status = hipDeviceSynchronize();
        if (status == hipSuccess) {
            status = hipMemcpy(&c, d_c, sizeof(float), hipMemcpyDeviceToHost);
        }
    }
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipblasDestroy(blas);
    if (blas_status != HIPBLAS_STATUS_SUCCESS) {
        return fail_blas("hipblasSgemm", blas_status);
    }
    if (status != hipSuccess) {
        return fail("hipBLAS synchronization", status);
    }
    if (c != 42.0f) {
        std::fprintf(stderr, "HIP runtime smoke test: invalid hipBLAS result %.3f\n", c);
        return 4;
    }

    std::printf("HIP + hipBLAS runtime smoke test passed on device %d: %s (%s)\n",
            device, selected.name, selected.gcnArchName);
    return 0;
}
