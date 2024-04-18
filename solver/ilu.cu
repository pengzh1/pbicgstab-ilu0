#include <algorithm>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <iomanip>
#include <thread>
#include <string>
#include <sstream>

#ifndef WARP_SIZE
#define WARP_SIZE   32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK  32
#endif


bool ludebug = true;

void cudaCheckError3() {
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::stringstream _error;
        _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        throw 999;
    }
}


const int maxGrid = 65536;


__global__ void findDiag(const int nrows, int *const __restrict__ diag_ptrs,
                         const int *const __restrict__ row_ptrs,
                         const int *const __restrict__ col_idxs) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= nrows || col_idxs[row_ptrs[gid]] > gid || col_idxs[row_ptrs[gid + 1] - 1] < gid) {
        return;
    }
    for (int i = row_ptrs[gid]; i < row_ptrs[gid + 1]; i += 1) {
        if (col_idxs[i] == gid) {
            diag_ptrs[gid] = i;
            break;
        }
        if (col_idxs[i] > gid) {
            break;
        }
    }
}

__global__ void findILU0(const int nrows, const int nnz, volatile double *const values,
                         const uint32_t *__restrict__ dependencies,
                         volatile bool *const ready, const int dep_size, const int dep_sub_size) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= nnz) {
        return;
    }
    const int start = gid * (dep_size / dep_sub_size) + 1;
    const int end = gid * (dep_size / dep_sub_size) + dependencies[start - 1];
    const bool has_diag_dependency = ((end + 1 - start) % 2 == 1 && start <= end);
    int diag_loc = -1;
    if (has_diag_dependency) {
        diag_loc = dependencies[end];
    }
    int current = start;
    double diag_value = 1;
    double u_val;
    double l_val;
    double sum = 0;
    bool u_flag = false;
    bool l_flag = false;
    bool finished = false;
    int l_loc, u_loc;
    bool next = false;
    while (!finished) {
        if ((has_diag_dependency && current <= end - 2) ||
            (!has_diag_dependency && current <= end - 1)) {
            if (!next) {
                l_loc = dependencies[current];
                u_loc = dependencies[current + 1];
                next = true;
            }
            if (!l_flag && ready[l_loc]) {
                l_val = values[l_loc];
                l_flag = true;
            }
            if (!u_flag && ready[u_loc]) {
                u_val = values[u_loc];
                u_flag = true;
            }
            if (l_flag && u_flag) {
                sum += l_val * u_val;
                current += 2;
                l_flag = false;
                u_flag = false;
                next = false;
            }
        }
        if (has_diag_dependency && current == end) {
            if (ready[diag_loc]) {
                diag_value = values[diag_loc];
                current++;
            }
        }
        if (current >= end + 1) {
            values[gid] = (values[gid] - sum) / diag_value;
            __threadfence();
            ready[gid] = true;
            finished = true;
        }
    }
}


__global__
void analyseMatrix(uint32_t *__restrict__ dependencies,
                   const int *__restrict__ diag_ptrs,
                   const int *__restrict__ h_csrRowPtr,
                   const int *__restrict__ h_csrColIdx,
                   const int m, const int dep_size, const int dep_sub_size) {
    const int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= m) {
        return;
    }
    uint32_t row_index = global_id;
    const int row_start = h_csrRowPtr[row_index];
    const int row_end = h_csrRowPtr[row_index + 1];
    for (int loc = row_start; loc < row_end; loc++) {
        uint32_t startIdx = loc * (dep_size / dep_sub_size);
        uint32_t dep = 1;
        int num_dependencies = 0;
        const int col_index = h_csrColIdx[loc];
        int k_max = row_index - 1;
        if (row_index > col_index) {
            k_max = col_index - 1;
        }
        for (int maybe_l_loc = row_start;
             maybe_l_loc < loc; maybe_l_loc++) {
            const int k = h_csrColIdx[maybe_l_loc];
            if (k > k_max) {
                break;
            }
            //find corresponding u at position k,col_index
            for (int maybe_u_loc = h_csrRowPtr[k]; maybe_u_loc < h_csrRowPtr[k + 1]; maybe_u_loc++) {
                if (h_csrColIdx[maybe_u_loc] == col_index) {
                    dependencies[startIdx + dep] = maybe_l_loc;
                    dependencies[startIdx + dep + 1] = maybe_u_loc;
                    dep += 2;
                    num_dependencies += 2;
                }
            }
        }
        if (row_index > col_index) {
            const int diag_loc = diag_ptrs[col_index];
            dependencies[startIdx + dep] = diag_loc;
            dep++;
            num_dependencies++;
        }
        dependencies[startIdx] = dep - 1;
    }
}

int analyseMatrixByGPU(uint32_t *d_dep,
                       int *d_diag_ptrs,
                       const int *d_csrRowPtr,
                       const int *d_csrColIdx,
                       const int m,
                       const int nnnz, const int dep_size, const int dep_sub_size) {
    const int nrows = m;
    int cur = 2 * nnnz;
    int dep = 0;
    time_t g1 = clock();
    // Matrix L
    // 分析矩阵元素LU分解依赖
    // 1.取对角元素位次
    findDiag<<<ceil(m / 1024) + 1, 1024 >>>(nrows, d_diag_ptrs, d_csrRowPtr,
                                            d_csrColIdx);
    cudaDeviceSynchronize();
    time_t g2 = clock();
    int *d_id_extractor;
    cudaMalloc((void **) &d_id_extractor, sizeof(int));
    cudaMemset(d_id_extractor, 0, sizeof(int));
    // 2.取所有元素的依赖元素
    analyseMatrix<<<ceil(m / 1024) + 1, 1024>>>(d_dep, d_diag_ptrs, d_csrRowPtr, d_csrColIdx,
                                                m, dep_size, dep_sub_size);
//    cudaCheckError3();
    time_t g3 = clock();

    int lastIndex = 0;
    time_t g4 = clock();
    time_t g5 = clock();
    // 打印LU分解预分析耗时统计
    if (ludebug) {
        printf("gTime %ld %ld %ld %ld \n", (g2 - g1) / (CLOCKS_PER_SEC / 1000), (g3 - g2) / (CLOCKS_PER_SEC / 1000),
               (g4 - g3) / (CLOCKS_PER_SEC / 1000), (g5 - g4) / (CLOCKS_PER_SEC / 1000));
    }
    return lastIndex;
}


void ILU0_MEGA(const int *d_csrRowPtr,
               const int *d_csrColIdx,
               double *d_csrVal,
               const int m, // rows
               const int nnnz, const int dep_size, const int dep_sub_size) {
    uint32_t *dependencies;
//    int *depcp;
    int *diag_ptrs;
    cudaMalloc((void **) &dependencies, (m + 1) * dep_size * sizeof(uint32_t));
//    cudaMalloc((void **) &depcp, (m + 1) * dep_sub_size * sizeof(int));
    cudaMalloc((void **) &diag_ptrs, m * sizeof(int));
    cudaMemset(dependencies, 0, (m + 1) * dep_sub_size * sizeof(int));
    cudaMemset(diag_ptrs, 0, m * sizeof(int));
    time_t ilu1 = clock();
    // 分析矩阵元素LU分解依赖
    // 1.取对角元素位次
    // 2.取所有元素的依赖元素
    analyseMatrixByGPU(dependencies, diag_ptrs, d_csrRowPtr, d_csrColIdx, m,
                       nnnz, dep_size, dep_sub_size);
    cudaDeviceSynchronize();
    time_t ilu2 = clock();

    dim3 block(1024);
    int grid_dim = ceil(
            (double) nnnz / (double) 1024) + 1;
    if (grid_dim > maxGrid) {
        grid_dim = maxGrid;
    }
    dim3 grid(grid_dim);
    bool *ready = nullptr;
    cudaMalloc((void **) &ready, nnnz * sizeof(bool));
    cudaMemset(ready, false, nnnz * sizeof(bool));
    cudaDeviceSynchronize();
    time_t ilu3 = clock();
    int *d_id_extractor;
    cudaMalloc((void **) &d_id_extractor, sizeof(int));
    cudaMemset(d_id_extractor, 0, sizeof(int));
    // 根据各个元素的依赖关系
    // 按照ILU-0算法进行计算LU分解后的矩阵
    findILU0<<< grid, block >>>(m, nnnz,
                                d_csrVal,
                                dependencies, ready, dep_size, dep_sub_size);
    cudaCheckError3();
    time_t ilu4 = clock();
//    cudaFree(dependencies);
//    cudaFree(ready);
    cudaCheckError3();
    cudaDeviceSynchronize();
    time_t ilu5 = clock();
    // 打印耗时统计
    if (ludebug) {
        printf("iluTime %ld %ld %ld %ld \n", (ilu2 - ilu1) / (CLOCKS_PER_SEC / 1000),
               (ilu3 - ilu2) / (CLOCKS_PER_SEC / 1000), (ilu4 - ilu3) / (CLOCKS_PER_SEC / 1000),
               (ilu5 - ilu4) / (CLOCKS_PER_SEC / 1000));
    }
}