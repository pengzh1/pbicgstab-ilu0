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
#include "ilu.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE   32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK  32
#endif

void cudaCheckError2() {
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::stringstream _error;
        _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        throw 999;
    }
}

void checkCudaError(cudaError_t e) {
    if (e != cudaSuccess) {
        std::stringstream _error;
        _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        cudaDeviceSynchronize();
        throw "xxx";
    }
}

void setUpDescriptor(cusparseMatDescr_t &descrA, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase) {
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, matrixType);
    cusparseSetMatIndexBase(descrA, indexBase);
}

__global__ void printArrF(const double *val) {
    printf("data is ");
    for (int i = 0; i < 20; i++) {
        printf("%f ", val[i]);
    }
    printf("\n");
}

__global__ void printArrI(const int *val) {
    printf("dataI is ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", val[i]);
    }
    printf("\n");
}

__global__
void spTrSolveL(const int *__restrict__ d_csrRowPtr,
                const int *__restrict__ d_csrColIdx,
                const double *__restrict__ d_csrVal,
                volatile bool *__restrict__ d_get_value,// 0*m
                const int m, // rows
                const double *__restrict__ d_b, // rhs
                double *d_x, // initVec
                int *d_id_extractor // 0
) {
//    const int global_id = atomicAdd(d_id_extractor, 1);
    const int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= m) {
        return;
    }
    int col, j, i;
    double xi;
    double left_sum = 0;
    i = global_id; // 3
    j = d_csrRowPtr[i];
    if (d_csrColIdx[j] > i) {
        return;
    }
    int end = d_csrRowPtr[i + 1] - 1;
    while (d_csrColIdx[end] > i) {
        end -= 1;
    }
    end += 1;
    while (j < end) { // 1,2
        col = d_csrColIdx[j];
        while (d_get_value[col] && j < end) {
            left_sum += d_csrVal[j] * d_x[col];
            j++;
            if (j < end) {
                col = d_csrColIdx[j];
            }
        }
        if (i == col || j == end) {
            xi = (d_b[i] - left_sum);
            d_x[i] = xi;
            __threadfence();
            d_get_value[i] = true;
            j++;
        }
    }
}

__global__
void spTrSolveU(const int *__restrict__ d_csrRowPtr,
                const int *__restrict__ d_csrColIdx,
                const double *__restrict__ d_csrVal,
                volatile bool *__restrict__ d_get_value,// 0*m
                const int m, // rows
                const double *__restrict__ d_b, // rhs
                double *d_x, // initVec
                int *d_id_extractor // 0
) {
//    int global_idx = atomicAdd(d_id_extractor, 1);
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_id = m - global_idx - 1;
    if (global_id < 0) {
        return;
    }
    int col, j, i;
    col = -1;
    double xi;
    double right_sum = 0;
    i = global_id; // 3
    j = d_csrRowPtr[i + 1] - 1;
    if(d_csrColIdx[j] < i) {
        return;
    }
    int end = d_csrRowPtr[i];
    while (d_csrColIdx[end] < i) {
        end += 1;
    }
    int itr = 0;
    while (j >= end && d_csrColIdx[j] >= i) { // 1,2
        itr += 1;
        col = d_csrColIdx[j];
        while (j >= end && d_csrColIdx[j] > i && d_get_value[col]) {
            right_sum += d_csrVal[j] * d_x[col];
            j--;
            if (j >= end && d_csrColIdx[j] > i) {
                col = d_csrColIdx[j];
            }
        }
        if (i == col || j == end) {
            xi = (d_b[i] - right_sum) / (d_csrVal[j]);
            d_x[i] = xi;
            __threadfence();
            d_get_value[i] = true;
            j--;
        }
    }
}


int spTrSolve(const int *__restrict__ d_csrRowPtr,
              const int *__restrict__ d_csrColIdx,
              const double *__restrict__ d_csrVal,
              const int m, // rows
              const int nnz, // nnz for L
              const double *__restrict__ d_b, // rhs
              double *d_x,// initVec,
              bool isL
) {

    //get_value
    bool *d_get_value;
    cudaMalloc((void **) &d_get_value, (m) * sizeof(bool));
    cudaMemset(d_get_value, false, sizeof(bool) * m);
    // step 5: solve L*y = x
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;;
    int num_blocks = ceil((double) m / (double) (num_threads));
    int *d_id_extractor;
    cudaMalloc((void **) &d_id_extractor, sizeof(int));
    cudaMemset(d_x, 0, sizeof(double) * m);
    cudaMemset(d_id_extractor, 0, sizeof(int));
    if (isL) {
        spTrSolveL<<< num_blocks, num_threads >>>
                (d_csrRowPtr, d_csrColIdx, d_csrVal,
                 d_get_value, m, d_b, d_x, d_id_extractor);
    } else {
        spTrSolveU<<< num_blocks, num_threads >>>
                (d_csrRowPtr, d_csrColIdx, d_csrVal,
                 d_get_value, m, d_b, d_x, d_id_extractor);
    }
    cudaCheckError2();
    cudaDeviceSynchronize();


    return 0;

}

void setDesc(cusparseMatDescr_t &descrLU, cusparseMatrixType_t matrixType,
             cusparseIndexBase_t indexBase, cusparseFillMode_t fillMode,
             cusparseDiagType_t diagType) {
    cusparseCreateMatDescr(&descrLU);
    cusparseSetMatType(descrLU, matrixType);
    cusparseSetMatIndexBase(descrLU, indexBase);
    cusparseSetMatFillMode(descrLU, fillMode);
    cusparseSetMatDiagType(descrLU, diagType);
}


void checkSpError(cusparseStatus_t error) {
    cudaDeviceSynchronize();
    switch (error) {
        case CUSPARSE_STATUS_SUCCESS:
            break;
        default:
            throw "Sp ERROR";
    }
}

void memQuery(csrilu02Info_t &infoA, csrsv2Info_t &infoL, csrsv2Info_t &infoU,
              cusparseHandle_t cusparseHandle, const int n, const int nnz,
              cusparseMatDescr_t &descrA, cusparseMatDescr_t &descrL, cusparseMatDescr_t &descrU,
              double *d_A, const int *d_A_RowPtr, const int *d_A_ColInd,
              cusparseOperation_t matrixOperation, void **pBuffer) {
    cusparseCreateCsrilu02Info(&infoA);
    cusparseCreateCsrsv2Info(&infoL);
    cusparseCreateCsrsv2Info(&infoU);

    int pBufferSize_M, pBufferSize_L, pBufferSize_U;
    cusparseDcsrilu02_bufferSize(cusparseHandle, n, nnz, descrA, d_A, d_A_RowPtr,
                                 d_A_ColInd, infoA, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(cusparseHandle, matrixOperation, n, nnz, descrL,
                               d_A, d_A_RowPtr, d_A_ColInd, infoL, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(cusparseHandle, matrixOperation, n, nnz, descrU,
                               d_A, d_A_RowPtr, d_A_ColInd, infoU, &pBufferSize_U);

    int pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));

    checkCudaError(cudaMalloc((void **) pBuffer, pBufferSize));
}


void spAnalyze(csrilu02Info_t &infoA, csrsv2Info_t &infoL,
               csrsv2Info_t &infoU, cusparseHandle_t cusparseHandle, const int N,
               const int nnz, cusparseMatDescr_t descrA, cusparseMatDescr_t &descrL,
               cusparseMatDescr_t &descrU, double *d_A, const int *d_A_RowPtr,
               const int *d_A_ColInd, cusparseOperation_t matrixOperation,
               cusparseSolvePolicy_t solvePolicy1, cusparseSolvePolicy_t solvePolicy2,
               void *pBuffer) {
    int structural_zero;
    time_t t1 = clock();


    checkSpError(cusparseDcsrilu02_analysis(cusparseHandle, N, nnz, descrA, d_A, d_A_RowPtr,
                                            d_A_ColInd, infoA, solvePolicy1, pBuffer));

    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(cusparseHandle, infoA, &structural_zero);

    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d, %d) is missing\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(cusparseHandle, matrixOperation, N, nnz, descrL,
                             d_A, d_A_RowPtr, d_A_ColInd, infoL, solvePolicy1, pBuffer);
    cusparseDcsrsv2_analysis(cusparseHandle, matrixOperation, N, nnz, descrU,
                             d_A, d_A_RowPtr, d_A_ColInd, infoU, solvePolicy2, pBuffer);
    cudaDeviceSynchronize();
//    printf("luCost%ld", (clock() - t1) / (CLOCKS_PER_SEC / 1000));
}

__global__ void printStrF() {

}

void setUpMatrix(cusparseHandle_t &cusparseHandle, cusparseMatDescr_t &descrA,
                 cusparseMatDescr_t &descrL, cusparseMatDescr_t &descrU, csrilu02Info_t &infoA,
                 csrsv2Info_t &infoL, csrsv2Info_t &infoU, int n, int nnz, double *valACopy,
                 const int *rowPtr, const int *colInd, void **pBuffer) {
    setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO);
    setDesc(descrL, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_UNIT);
    setDesc(descrU, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // Step 2: Query how much memory used in LU factorization and the two following system inversions.
    memQuery(infoA, infoL, infoU, cusparseHandle, n, nnz, descrA, descrL, descrU,
             valACopy, rowPtr, colInd, CUSPARSE_OPERATION_NON_TRANSPOSE, pBuffer);
//     Step 3: Analyze the three problems: LU factorization and the two following system inversions.
//    spAnalyze(infoA, infoL, infoU, cusparseHandle, n, nnz, descrA, descrL, descrU,
//              valACopy, rowPtr, colInd, CUSPARSE_OPERATION_NON_TRANSPOSE,
//              CUSPARSE_SOLVE_POLICY_NO_LEVEL, CUSPARSE_SOLVE_POLICY_NO_LEVEL, *pBuffer);
    cudaCheckError2();
}

// TODO perfFix
void spNewMV(cusparseHandle_t handle,
             cusparseOperation_t transA,
             int m,
             int n,
             int nnz,
             const double *alpha,
             const cusparseMatDescr_t descrA,
             const double *csrValA,
             const int *csrRowPtrA,
             const int *csrColIndA,
             const double *x,
             const double *beta,
             double *y) {
    time_t mv0 = clock();
    double one = 1, nega_one = -1, zero = 0;
    int *rows = const_cast<int *>(csrRowPtrA);
    int *cols = const_cast<int *>(csrColIndA);
    double *vals = const_cast<double *>(csrValA);
    cusparseSpMatDescr_t matA_descr;
    cusparseDnVecDescr_t vecX_descr;
    cusparseDnVecDescr_t vecY_descr;
    time_t mv1 = clock();
    checkSpError(cusparseCreateDnVec(&vecX_descr, m, const_cast<double *>(x), CUDA_R_64F));
    checkSpError(cusparseCreateDnVec(&vecY_descr, m, const_cast<double *>(y), CUDA_R_64F));
    checkSpError(cusparseCreateCsr(&matA_descr, m, n, nnz, const_cast<int *>(rows), const_cast<int *>(cols),
                                   const_cast<double *>(vals),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    size_t bufferSize = 0;
    checkSpError(cusparseSpMV_bufferSize(handle, transA, alpha, matA_descr, vecX_descr, beta, vecY_descr, CUDA_R_64F,
                                         CUSPARSE_CSRMV_ALG1, &bufferSize));
    cudaCheckError2();
    void *dBuffer = NULL;
    cudaMalloc(&dBuffer, bufferSize);
    cudaDeviceSynchronize();
    time_t mv2 = clock();
    checkSpError(cusparseSpMV(handle, transA, alpha, matA_descr, vecX_descr, beta, vecY_descr, CUDA_R_64F,
                              CUSPARSE_SPMV_CSR_ALG2, dBuffer));
    cudaDeviceSynchronize();
    cudaCheckError2();
    time_t mv3 = clock();
    cusparseDestroySpMat(matA_descr);
    cusparseDestroyDnVec(vecX_descr);
    cusparseDestroyDnVec(vecY_descr);
    cudaDeviceSynchronize();
    cudaCheckError2();
    time_t mv4 = clock();
//    printf("mvTime %ld %ld %ld %ld \n",
//           (mv4 - mv3) / (CLOCKS_PER_SEC / 1000), (mv3 - mv2) / (CLOCKS_PER_SEC / 1000),
//           (mv2 - mv1) / (CLOCKS_PER_SEC / 1000),
//           (mv1 - mv0) / (CLOCKS_PER_SEC / 1000));
}

void spSolverBiCGStab(int n, int nnz, const double *valA, const int *rowPtr, const int *colInd,
                      const double *b, double *x, double tol, cusparseHandle_t cusparseHandle,
                      cublasHandle_t cublasHandle) {
    time_t solve_start = clock();
    // Create descriptors for A, L and U.
    cusparseMatDescr_t descrA, descrL, descrU;

    // Create ILU and SV info for A, L and U.
    csrilu02Info_t infoA;
    csrsv2Info_t infoL, infoU;

    // Create a copy of A for incomplete LU decomposition.
    // This copy will be modified in the solving process.
    double *valACopy;
    cudaMalloc((void **) &valACopy, nnz * sizeof(double));
    cudaMemcpy(valACopy, valA, nnz * sizeof(double), cudaMemcpyDeviceToDevice);

    // Incomplete LU.
    time_t solve_start1 = clock();
    void *pBuffer;
    setUpMatrix(cusparseHandle, descrA, descrL, descrU, infoA, infoL, infoU, n, nnz, valACopy, rowPtr, colInd,
                &pBuffer);
    int *diag_info = nullptr;
    cudaMalloc((void **) &diag_info, sizeof(int) * n);
    cudaCheckError2();
    ILU0_MEGA(rowPtr, colInd, valACopy,
              n,
              nnz);
    cudaDeviceSynchronize();
    time_t solve_start2 = clock();
    double *r;
    checkCudaError(cudaMalloc((void **) &r, n * sizeof(double)));
    double *rw;
    cudaMalloc((void **) &rw, n * sizeof(double));
    double *p;
    cudaMalloc((void **) &p, n * sizeof(double));
    double *ph;
    cudaMalloc((void **) &ph, n * sizeof(double));
    double *t;
    cudaMalloc((void **) &t, n * sizeof(double));
    double *q;
    cudaMalloc((void **) &q, n * sizeof(double));
    double *s;
    checkCudaError(cudaMalloc((void **) &s, n * sizeof(double)));
    time_t solve_start3 = clock();

    double one = 1, nega_one = -1, zero = 0;
    double alpha, negalpha, beta, omega, nega_omega;
    double temp1, temp2;
    double rho = 0.0, rhop;
    double nrmr0;
    double nrmr;
    int niter = 0;

    // Initial guess x0 (all zeros here).
//    cublasDscal_v2(cublasHandle, n, &zero, x, 1);
    // 1: compute the initial residual r = b - A * x0.
    spNewMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &nega_one, descrA, valA, rowPtr,
            colInd, x, &zero, r);
    cudaCheckError2();

    cublasDaxpy_v2(cublasHandle, n, &one, b, 1, r, 1);
    // 2: copy r into rw and p.
    cublasDcopy_v2(cublasHandle, n, r, 1, rw, 1);
    cublasDcopy_v2(cublasHandle, n, r, 1, p, 1);
    time_t solve_start4 = clock();
    cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr0);
    printf("initNRMR %f \n", nrmr0);
    // Repeat until convergence.
    while (true) {
        printf("niter %d ms %ld\n", niter, clock() / (CLOCKS_PER_SEC / 1000));
        time_t it0 = clock();
        rhop = rho;
        cublasDdot_v2(cublasHandle, n, rw, 1, r, 1, &rho);
        if (niter > 0) {
            beta = (rho / rhop) * (alpha / omega);
            //  p = r + beta * (p - omega * v)
            cublasDaxpy_v2(cublasHandle, n, &nega_omega, q, 1, p, 1);  // p += -omega * v
            cublasDscal_v2(cublasHandle, n, &beta, p, 1);  // p *= beta
            cublasDaxpy_v2(cublasHandle, n, &one, r, 1, p, 1);  // p += 1 * r
        }
        cudaDeviceSynchronize();
        time_t it1 = clock();
        spTrSolve(rowPtr, colInd, valACopy, n, nnz, p, t, true);
        cudaDeviceSynchronize();
        time_t it2 = clock();
        spTrSolve(rowPtr, colInd, valACopy, n, nnz, t, ph, false);
        spNewMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, valA, rowPtr, colInd,
                ph, &zero, q);
        cudaDeviceSynchronize();
        time_t it3 = clock();
        cublasDdot_v2(cublasHandle, n, rw, 1, q, 1, &temp1);
        alpha = rho / temp1;
        negalpha = -alpha;
        cublasDaxpy_v2(cublasHandle, n, &negalpha, q, 1, r, 1);
        cublasDaxpy_v2(cublasHandle, n, &alpha, ph, 1, x, 1);
        cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr);
        cudaDeviceSynchronize();
        time_t it4 = clock();
        if ((nrmr / nrmr0) < tol) {
            std::cout << std::setprecision(12) << nrmr / nrmr0 << " " << nrmr << " NRMR \n";
            break;
        }
        spTrSolve(rowPtr, colInd, valACopy, n, nnz, r, t, true);
        cudaDeviceSynchronize();
        time_t it5 = clock();
        spTrSolve(rowPtr, colInd, valACopy, n, nnz, t, s, false);
        spNewMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descrA, valA, rowPtr, colInd,
                s, &zero, t);
        cublasDdot_v2(cublasHandle, n, t, 1, r, 1, &temp1);
        cudaDeviceSynchronize();
        time_t it6 = clock();
        cublasDdot_v2(cublasHandle, n, t, 1, t, 1, &temp2);
        omega = temp1 / temp2;
        nega_omega = -omega;
        cublasDaxpy_v2(cublasHandle, n, &omega, s, 1, x, 1);
        cublasDaxpy_v2(cublasHandle, n, &nega_omega, t, 1, r, 1);
        cublasDnrm2_v2(cublasHandle, n, r, 1, &nrmr);
        cudaDeviceSynchronize();
        time_t it7 = clock();
        printf("itTime %ld %ld %ld %ld %ld %ld %ld \n", (it7 - it6) / (CLOCKS_PER_SEC / 1000),
               (it6 - it5) / (CLOCKS_PER_SEC / 1000),
               (it5 - it4) / (CLOCKS_PER_SEC / 1000),
               (it4 - it3) / (CLOCKS_PER_SEC / 1000),
               (it3 - it2) / (CLOCKS_PER_SEC / 1000),
               (it2 - it1) / (CLOCKS_PER_SEC / 1000),
               (it1 - it0) / (CLOCKS_PER_SEC / 1000));
        if ((nrmr / nrmr0) < tol) {
            std::cout << std::setprecision(12) << nrmr / nrmr0 << nrmr << " NRMR \n";
            break;
        }
        niter++;
    }
    time_t solve_start5 = clock();
    time_t solve_start6 = clock();
    time_t solve_end = clock();
    printf("solveTime %ld %ld %ld %ld %ld %ld %ld %ld\n",
           (solve_end - solve_start) / (CLOCKS_PER_SEC / 1000),
           (solve_end - solve_start6) / (CLOCKS_PER_SEC / 1000),
           (solve_start6 - solve_start5) / (CLOCKS_PER_SEC / 1000),
           (solve_start5 - solve_start4) / (CLOCKS_PER_SEC / 1000),
           (solve_start4 - solve_start3) / (CLOCKS_PER_SEC / 1000),
           (solve_start3 - solve_start2) / (CLOCKS_PER_SEC / 1000),
           (solve_start2 - solve_start1) / (CLOCKS_PER_SEC / 1000),
           (solve_start1 - solve_start) / (CLOCKS_PER_SEC / 1000));
}
