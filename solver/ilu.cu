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


#ifndef DEP_SIZE
#define DEP_SIZE   128
#endif

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
    if (gid >= nrows) {
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
                         const int *const dependencies, const int *const nz_ptrs,
                         int *rowMap, int *colSortMap,
                         volatile bool *const ready, int *idx) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
//    const int gid = atomicAdd(idx, 1);
    if (gid == 0) {
        printf("x");
    }
    if (gid >= nnz) {
        return;
    }
    int itr = 0;
    const int loc = gid;
//        const int start = nz_ptrs[loc];
//        const int end = nz_ptrs[loc + 1] - 1;
    int rowStart = rowMap[loc] * DEP_SIZE;
    int myStart = (nz_ptrs[loc] - nz_ptrs[loc - colSortMap[loc]]);
    const int start = rowStart + myStart + 1;
    const int end = start + nz_ptrs[loc + 1] - nz_ptrs[loc] - 1;
//    if (end > nnz) {
//        printf("startMap:%d %d %d %d %d %d %d\n", gid, nz_ptrs[loc], nz_ptrs[loc + 1] - 1, start, end,
//               nz_ptrs[loc + 1], nz_ptrs[loc]);
//    }
    const bool has_diag_dependency = ((end + 1 - start) % 2 == 1 && start <= end);
    int current = start;
    double diag_value = 1;
    double u_val;
    double l_val;
    double sum = 0;
    bool u_flag = false;
    bool l_flag = false;
    bool finished = false;
//    printf("startFOr:%d %d %d %d\n", gid, has_diag_dependency, start, end);
    while (!finished) {
        itr += 1;
        if ((has_diag_dependency == true && current <= end - 2) ||
            (has_diag_dependency == false && current <= end - 1)) {
            const int l_loc = dependencies[current];
            const int u_loc = dependencies[current + 1];
            if (l_flag == false && ready[l_loc] == true) {
                l_val = values[l_loc];
                l_flag = true;
            }
            if (u_flag == false && ready[u_loc] == true) {
                u_val = values[u_loc];
                u_flag = true;
            }
            if (l_flag == true && u_flag == true) {
                sum += l_val * u_val;
                current += 2;
                l_flag = false;
                u_flag = false;
            }
        }
        if (has_diag_dependency == true && current == end) {
//            printf("startFOr:%d %d %d %d\n", gid, has_diag_dependency, start, end);
            const int diag_loc = dependencies[end];
            if (diag_loc >= nnz) {
                printf("dange6");
            }
            if (ready[diag_loc] == true) {
                diag_value = values[diag_loc];
                current++;
            }
        }
//        if (current == end + 1) {
//            printf("Finish :%d %d %d %d\n", gid, has_diag_dependency, start, end);
//        }
        if (current >= end + 1) {
//            printf("Finish %d", loc);
            values[loc] = (values[loc] - sum) / diag_value;
            __threadfence();
            ready[loc] = true;
//            __threadfence();
            finished = true;
//            return;
        }
    }
}

// single block acc
__global__
// 351864
void sumList(int len, int *nz_ptrs) {
    const int id = threadIdx.x;
    int perCur = len / blockDim.x + 1;
    const int start = id * perCur;
    int end = (id + 1) * perCur;
    for (int i = start; i < len - 1 && i < end - 1; i++) {
//        printf("get xe %d %d %d %d %d\n", id, start, end, i, len);
        nz_ptrs[i + 1] += nz_ptrs[i];
//        printf("get xe2 %d %d %d %d\n", id, start, end, i);
    }
//    printf("get stbc %d %d %d\n", id, start, end);
    __syncthreads();
//    printf("get stbcAfter %d %d %d\n", id, start, end);
    if (start < len) {
//        printf("get stbcAfter %d %d %d\n", id, start, end);
        int sum = 0;
        for (int i = 0; i < id; i++) {
            sum += nz_ptrs[(i + 1) * perCur - 1];
        }
        for (int i = start; i < len && i < end; i++) {
            nz_ptrs[i] += sum;
        }
    }
}

__global__
void refine(int m, int *dep, int *lastIndex) {
    int lastIdx = 0;
    printf("getLas01 %d %d", m, *lastIndex);
    for (int i = 0; i < m; i++) {
//        printf("getLas01 %d %d %d", m, *lastIndex, dep[i * 64]);
        int l = dep[i * DEP_SIZE];
        for (int j = 0; j < l; j++) {
            dep[lastIdx++] = dep[i * DEP_SIZE + j + 1];
        }
    }
//    printf("getLast1 %d %d", lastIdx, *lastIndex);
    *lastIndex = lastIdx;
    dep[0] = lastIdx;
//    printf("getLast %d %d", lastIdx, *lastIndex);
}

__global__
void refine2(int *dep, int *dep2, int m, int *lastIdx) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid > m) {
        return;
    }
    int start = 0;
    for (int i = 0; i < gid; i++) {
        start += dep2[i * DEP_SIZE];
    }
    int size = dep2[gid * DEP_SIZE];
    __syncthreads();
    for (int i = 0; i < size; i++) {
        dep[start + i] = dep2[gid * DEP_SIZE + i + 1];
    }
    atomicAdd(lastIdx, size);
}

// single block acc

__global__
void analyseMatrix(int *dependencies,
                   int *nz_ptrs,
                   int *diag_ptrs,
                   const int *h_csrRowPtr,
                   const int *h_csrColIdx,
                   const int m,
                   const int nnnz,
                   int *d_id_extractor) {
    const int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= m) {
        return;
    }
    nz_ptrs[0] = 0;
    int row_index = global_id;
    int startIdx = row_index * DEP_SIZE;
    int dep = 1;
    const int row_start = h_csrRowPtr[row_index];
    const int row_end = h_csrRowPtr[row_index + 1];
    for (int loc = row_start; loc < row_end; loc++) {
        int num_dependencies = 0;
        const int col_index = h_csrColIdx[loc];
        int k_max = row_index - 1;
        if (row_index > col_index) {
            k_max = col_index - 1;
        }
        for (int maybe_l_loc = row_start;
             maybe_l_loc < loc; maybe_l_loc++) //use loc instead of row_end as the matrix is sorted
        {
            const int k = h_csrColIdx[maybe_l_loc];
            if (k > k_max) {
                continue;
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
            const int diag_loc = diag_ptrs[col_index]; //diag_ptrs[col_index] has correct value as it has been found when doing stuff for previous rows as col_index < row_index here
            dependencies[startIdx + dep] = diag_loc;
            __threadfence();
            dep++;
            num_dependencies++;
        }
        nz_ptrs[loc + 1] = num_dependencies;
    }
    dependencies[startIdx] = dep - 1;
}

int analyseMatrixByGPU(int *d_dep,
                       int *d_ptrs,
                       int *d_diag_ptrs,
                       const int *d_csrRowPtr,
                       const int *d_csrColIdx,
                       const int m,
                       const int nnnz) {
    const int nrows = m;
    int cur = 2 * nnnz;
    int dep = 0;
    time_t g1 = clock();
    // Matrix L
    findDiag<<<ceil(m / 1024) + 1, 1024 >>>(nrows, d_diag_ptrs, d_csrRowPtr,
                                            d_csrColIdx);
    cudaDeviceSynchronize();
    time_t g2 = clock();
    int *d_id_extractor;
    cudaMalloc((void **) &d_id_extractor, sizeof(int));
    cudaMemset(d_id_extractor, 0, sizeof(int));

    analyseMatrix<<<ceil(m / 1024) + 1, 1024>>>(d_dep, d_ptrs, d_diag_ptrs, d_csrRowPtr, d_csrColIdx,
                                                m,
                                                nnnz,
                                                d_id_extractor);
//    cudaCheckError3();
    time_t g3 = clock();

    sumList<<<1, 1024>>>(nnnz + 1, d_ptrs);
    int lastIndex = 0;
    time_t g4 = clock();
    time_t g5 = clock();

    printf("gTime %ld %ld %ld %ld \n", (g2 - g1) / (CLOCKS_PER_SEC / 1000), (g3 - g2) / (CLOCKS_PER_SEC / 1000),
           (g4 - g3) / (CLOCKS_PER_SEC / 1000), (g5 - g4) / (CLOCKS_PER_SEC / 1000));
//    cudaCheckError3();
    return lastIndex;
}


void ILU0_MEGA(const int *d_csrRowPtr,
               const int *d_csrColIdx,
               double *d_csrVal,
               int *rowMap, int *colSortMap,
               const int m, // rows
               const int nnnz) {
    int *nz_ptrs;
    int *dependencies;
//    int *depcp;
    int *diag_ptrs;
    cudaMalloc((void **) &dependencies, (m + 1) * DEP_SIZE * sizeof(int));
//    cudaMalloc((void **) &depcp, (m + 1) * DEP_SIZE * sizeof(int));
    cudaMalloc((void **) &nz_ptrs, (nnnz + 1) * sizeof(int));
    cudaMemset(nz_ptrs, 0, (nnnz + 1) * sizeof(int));
    cudaMalloc((void **) &diag_ptrs, m * sizeof(int));
    cudaMemset(dependencies, 0, (m + 1) * DEP_SIZE * sizeof(int));
    cudaMemset(diag_ptrs, 0, m * sizeof(int));
    time_t ilu1 = clock();
    analyseMatrixByGPU(dependencies, nz_ptrs, diag_ptrs, d_csrRowPtr, d_csrColIdx, m,
                       nnnz);
//    int newDep = create_dependency_graph(dependencies_cpu, nz_ptrs_cpu, diag_ptrs_cpu, h_csrRowPtr, h_csrColIdx, m,
//                                         nnnz);
    cudaDeviceSynchronize();
    time_t ilu2 = clock();

    dim3 block(1024);
    int grid_dim = ceil(
            (double) nnnz / (double) 1024) + 1;


    if (grid_dim > maxGrid) {
        //std::cout << "\n Using max possible grid dim at line:"  << __LINE__ << "\n";
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
    findILU0 <<< grid, block >>>(m, nnnz,
                                 d_csrVal,
                                 dependencies, nz_ptrs, rowMap, colSortMap, ready,
                                 d_id_extractor);
    cudaCheckError3();
    time_t ilu4 = clock();

    cudaFree(dependencies);
    cudaFree(nz_ptrs);
    cudaFree(ready);
    cudaCheckError3();
    cudaDeviceSynchronize();
    time_t ilu5 = clock();
    printf("iluTime %ld %ld %ld %ld \n", (ilu2 - ilu1) / (CLOCKS_PER_SEC / 1000),
           (ilu3 - ilu2) / (CLOCKS_PER_SEC / 1000), (ilu4 - ilu3) / (CLOCKS_PER_SEC / 1000),
           (ilu5 - ilu4) / (CLOCKS_PER_SEC / 1000));

}