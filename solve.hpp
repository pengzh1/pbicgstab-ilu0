#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include <time.h>
#include <iomanip>
#include <thread>
#include <string>
#include <sstream>
#include "solver/bicgstab.cuh"
#include "solver/ilu.cuh"

#ifndef AMG_SOLVE_HPP
#define AMG_SOLVE_HPP

#endif //AMG_SOLVE_HPP

std::time_t getTimeStamp() {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
    //std::time_t timestamp = std::chrono::system_clock::to_time_t(tp);
    return timestamp;
}

std::tm *gettm(std::time_t timestamp) {
    std::time_t milli = timestamp/*+ (std::time_t)8*60*60*1000*/;//此处转化为东八区北京时间，如果是其它时区需要按需求修改
    auto mTime = std::chrono::milliseconds(milli);
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(mTime);
    auto tt = std::chrono::system_clock::to_time_t(tp);
    std::tm *now = std::gmtime(&tt);
    printf("%4d年%02d月%02d日 %02d:%02d:%02d.%d\n", now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour,
           now->tm_min, now->tm_sec, milli % 1000);
    return now;
}

void cudaCheckError() {
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        throw "Cuda failure";
    }
}

double tol = 1e-5;

void solve(int *rowPtr, int *colInd, double *csrData,
           double *b, double *x, int n, int nnz, double tol, cusparseHandle_t cusparseHandle,
           cublasHandle_t cublasHandle) {
    time_t solve_start = clock();
    int *gColInd, *gRowPtr;
    double *gCsrData, *gB, *gX;
    cudaMalloc((void **) &gColInd, nnz * sizeof(int));
    cudaMalloc((void **) &gRowPtr, (n + 1) * sizeof(int));
    cudaMalloc((void **) &gCsrData, nnz * sizeof(double));
    cudaMalloc((void **) &gB, n * sizeof(double));
    cudaMalloc((void **) &gX, n * sizeof(double));
    cudaMemcpyAsync(gColInd, colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gRowPtr, rowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gCsrData, csrData, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gB, b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gX, x, n * sizeof(double), cudaMemcpyHostToDevice);
    time_t solve_start1 = clock();
    time_t solve_start2 = clock();
    cudaCheckError();
    // Solve Ax = b for x.
    spSolverBiCGStab(n, nnz, gCsrData, gRowPtr, gColInd, gB, gX, tol, cusparseHandle,
                     cublasHandle);
    time_t stab_end = clock();
    // Copy x back to CPU.
    cudaMemcpy(x, gX, n * sizeof(double), cudaMemcpyDeviceToHost);
    time_t solve_end = clock();
}

