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
#include "bicgstab.cuh"
#include "ilu.cuh"

void cudaCheckError() {
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        throw "Cuda failure";
    }
}

void solve(int *rowPtr, int *colInd, double *csrData, int *rowMap, int *colSortMap,
           double *b, double *x, int n, int nnz, double tol, cusparseHandle_t cusparseHandle,
           cublasHandle_t cublasHandle) {
    time_t solve_start = clock();
    // Allocate GPU memory
    // ------------------------------------------
    // Copy CSR column indices to GPU.
    int *gColInd;
    // TODO perfFix，memInitAllocateSlow？
    cudaMalloc((void **) &gColInd, nnz * sizeof(int));
    cudaMemcpy(gColInd, colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);

    time_t solve_start1 = clock();
    // Copy CSR row offsets to GPU.
    int *gRowPtr;
    cudaMalloc((void **) &gRowPtr, (n + 1) * sizeof(int));
    cudaMemcpy(gRowPtr, rowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);


    // Copy CSR data array to GPU.
    double *gCsrData;
    cudaMalloc((void **) &gCsrData, nnz * sizeof(double));
    cudaMemcpy(gCsrData, csrData, nnz * sizeof(double), cudaMemcpyHostToDevice);

    int *gRowMap;
    cudaMalloc((void **) &gRowMap, nnz * sizeof(int));
    cudaMemcpy(gRowMap, rowMap, nnz * sizeof(int), cudaMemcpyHostToDevice);
    int *gColSortMap;
    cudaMalloc((void **) &gColSortMap, nnz * sizeof(int));
    cudaMemcpy(gColSortMap, colSortMap, nnz * sizeof(int), cudaMemcpyHostToDevice);
    time_t solve_start2 = clock();


    // Residual vector.
    double *gB;
    cudaMalloc((void **) &gB, n * sizeof(double));
    cudaMemcpy(gB, b, n * sizeof(double), cudaMemcpyHostToDevice);
    // Solution.
    double *gX;
    cudaMalloc((void **) &gX, n * sizeof(double));
    // ------------------------------------------
    cudaCheckError();
    // Solve Ax = b for x.
    spSolverBiCGStab(n, nnz, gCsrData, gRowPtr, gColInd, gRowMap, gColSortMap, gB, gX, tol, cusparseHandle,
                     cublasHandle);
    time_t stab_end = clock();

    // Copy x back to CPU.
    cudaMemcpy(x, gX, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(gX);
    cudaFree(gB);

    cudaFree(gCsrData);
    cudaFree(gColInd);
    cudaFree(gRowPtr);
    time_t solve_end = clock();
//    printf("memCpTime1 %ld %ld %ld %ld %ld %ld, solve time %ld,memRmTime %ld\n",
//           (stab_start - solve_start) / (CLOCKS_PER_SEC / 1000),
//           (stab_start - solve_start4) / (CLOCKS_PER_SEC / 1000),
//           (solve_start4 - solve_start3) / (CLOCKS_PER_SEC / 1000),
//           (solve_start3 - solve_start2) / (CLOCKS_PER_SEC / 1000),
//           (solve_start2 - solve_start1) / (CLOCKS_PER_SEC / 1000),
//           (solve_start1 - solve_start) / (CLOCKS_PER_SEC / 1000),
//           (stab_end - stab_start) / (CLOCKS_PER_SEC / 1000),
//           (solve_end - stab_end) / (CLOCKS_PER_SEC / 1000));
}

int solveMat(int nrow, int nnnz, int *rowPtr, int *colInd, double *csrData, int *rowMap, int *colSortMap, double *b,
             double *x) {
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    cusparseCreate(&cusparseHandle);
//    time_t solve_start4 = clock();
    // TODO perfFix
    cublasCreate_v2(&cublasHandle);
//    for (int p = 0; p < 2; p++) {
    time_t run_start = clock();
    // Vector size.
    int n = nrow;
    // None-Zero count.
    int nnz = nnnz;
    // Tolerance.
    double tol = 1e-3;
//        for (int xxx = 0; xxx < p; xxx++) {
//            tol = tol / 1000;
//        }
    time_t solve_start = clock();
    solve(rowPtr, colInd, csrData, rowMap, colSortMap, b, x, n, nnz, tol, cusparseHandle, cublasHandle);
    time_t solve_end = clock();
    for (int i = 0; i < 5; ++i) {
        std::cout << std::setprecision(12) << x[i] << " ";
    }
    std::cout << std::endl;
//        printf("runtime %ld %ld %ld %ld, solve time %ld\n", (solve_start - run_start) / (CLOCKS_PER_SEC / 1000),
//               (solve_start - run_start2) / (CLOCKS_PER_SEC / 1000),
//               (run_start2 - run_start1) / (CLOCKS_PER_SEC / 1000),
//               (run_start1 - run_start) / (CLOCKS_PER_SEC / 1000),
//               (solve_end - solve_start) / (CLOCKS_PER_SEC / 1000));
    std::cout << std::endl;
//        printArrIC(rowPtr);

//    }
    // Clean up.
    cusparseDestroy(cusparseHandle);
    cublasDestroy_v2(cublasHandle);
    return 0;
}

//int main(int argc, char *argv[]) {
//    std::ifstream in1;
//    in1.open("/home/featurize/data/mock3.mtx");
//    int tNrow = 3;
//    int *tRowPtr = new int[tNrow + 1]{};
//    for (int i = 0; i < tNrow + 1; ++i) {
//        in1 >> tRowPtr[i];
//    }
//    int tNnnz = tRowPtr[tNrow];
//    int *tColInd = new int[tNnnz]{};
//    double *tCsrData = new double[tNnnz]{};
//    int *tRowMap = new int[tNnnz];
//    int *tColSortMap = new int[tNnnz];
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tColInd[i];
//    }
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tCsrData[i];
//    }
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tRowMap[i];
//    }
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tColSortMap[i];
//    }
//    int nrow = tNrow;
//    int nnnz = tNnnz;
//    int *rowPtr = tRowPtr;
//    int *colInd = tColInd;
//    double *csrData = tCsrData;
////    int expectRow = 3;
////    if (expectRow != -1) {
////        nrow = expectRow;
////        rowPtr = new int[nrow + 1];
////        rowPtr[0] = 0;
////        for (int i = 1; i < nrow + 1; ++i) {
////            rowPtr[i] = tRowPtr[tNrow - expectRow + i] - tRowPtr[tNrow - expectRow];
////        }
////        nnnz = rowPtr[nrow];
////        colInd = new int[nnnz];
////        csrData = new double[nnnz];
////        for (int i = 0; i < nnnz; ++i) {
////            colInd[i] = tColInd[tNnnz - nnnz + i];
////        }
////        for (int i = 0; i < nnnz; ++i) {
////            csrData[i] = tCsrData[tNnnz - nnnz + i];
////        }
////    }
//
//    int *d_csrRowPtrL;
//    int *d_csrColIdx;
//    int *d_get_value;
//    int *d_rowmap;
//    int *d_colOrderMap;
//    double *d_csrValL;
//    double *d_b;
//    double *d_x;
//
//
//    // Matrix L
//    cudaMalloc((void **) &d_csrRowPtrL, (nrow + 1) * sizeof(int));
//    cudaMalloc((void **) &d_csrColIdx, nnnz * sizeof(int));
//    cudaMalloc((void **) &d_csrValL, nnnz * sizeof(double));
//
//    cudaMemcpy(d_csrRowPtrL, rowPtr, (nrow + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrColIdx, colInd, nnnz * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrValL, csrData, nnnz * sizeof(double), cudaMemcpyHostToDevice);
//
//    cudaMalloc((void **) &d_rowmap, nnnz * sizeof(int));
//    cudaMalloc((void **) &d_colOrderMap, nnnz * sizeof(int));
//
//    cudaMemcpy(d_csrRowPtrL, rowPtr, (nrow + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrColIdx, colInd, nnnz * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrValL, csrData, nnnz * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_rowmap, tRowMap, nnnz * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_colOrderMap, tColSortMap, nnnz * sizeof(int ), cudaMemcpyHostToDevice);
//
////    // Vector b
////    cudaMalloc((void **) &d_b, nrow * sizeof(double));
////    cudaMemcpy(d_b, b, nrow * sizeof(double), cudaMemcpyHostToDevice);
////
////    cudaMalloc((void **) &d_x, nrow * sizeof(double));
////    cudaMemcpy(d_x, x, nrow * sizeof(double), cudaMemcpyHostToDevice);
////    spTrSolve(d_csrRowPtrL, d_csrColIdx, d_csrValL, nrow, nnnz, d_b, d_x, false);
//    int *diag_info = nullptr;
//    cudaMalloc((void **) &diag_info, sizeof(int) * nrow);
////    printf("isNull %d",d_csrColIdx== nullptr);
////    find_locn_of_diag_elements(nrow, diag_info, d_csrRowPtrL,
////            d_csrColIdx);
//
//    ILU0_MEGA(d_csrRowPtrL, d_csrColIdx, d_csrValL, d_rowmap,d_colOrderMap,
//                         nrow, // rows
//                         nnnz);
//    cudaCheckError();
//
//    cudaMemcpy(csrData, d_csrValL, nnnz * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nnnz; i++) {
//        std::cout << std::setprecision(12) << csrData[i] << "\n";
//    }
//
//}

//int main(int argc, char *argv[]) {
//    std::ifstream in1;
//    in1.open("/home/featurize/data/mock2.mtx");
//    int tNrow = 351864;
//    int *tRowPtr = new int[tNrow + 1]{};
//    for (int i = 0; i < tNrow + 1; ++i) {
//        in1 >> tRowPtr[i];
//    }
//    int tNnnz = tRowPtr[tNrow];
//    int *tColInd = new int[tNnnz]{};
//    double *tCsrData = new double[tNnnz]{};
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tColInd[i];
//    }
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tCsrData[i];
//    }
//    double *tB = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        in1 >> tB[i];
//    }
//    double *x = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        x[i] = 0;
//    }
//    double *rx = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        in1 >> rx[i];
//    }
//    int nrow, nnnz;
//    int *rowPtr, *colInd;
//    double *csrData, *b;
//    int expectRow = 351864;
//    if (expectRow != -1) {
//        nrow = expectRow;
//        rowPtr = new int[nrow + 1];
//        rowPtr[0] = 0;
//        for (int i = 1; i < nrow + 1; ++i) {
//            rowPtr[i] = tRowPtr[tNrow - expectRow + i] - tRowPtr[tNrow - expectRow];
//        }
//        nnnz = rowPtr[nrow];
//        colInd = new int[nnnz];
//        csrData = new double[nnnz];
//        for (int i = 0; i < nnnz; ++i) {
//            colInd[i] = tColInd[tNnnz-nnnz+i];
//        }
//        for (int i = 0; i < nnnz; ++i) {
//            csrData[i] = tCsrData[tNnnz-nnnz+i];
//        }
//        b = new double[nrow];
//        for (int i = 0; i < nrow; ++i) {
//            b[i] = tB[tNrow-nrow+i];
//        }
//    }
//
//    int *d_csrRowPtrL;
//    int *d_csrColIdx;
//    int *d_get_value;
//    double *d_csrValL;
//    double *d_b;
//    double *d_x;
//
//
//    // Matrix L
//    cudaMalloc((void **) &d_csrRowPtrL, (nrow + 1) * sizeof(int));
//    cudaMalloc((void **) &d_csrColIdx, nnnz * sizeof(int));
//    cudaMalloc((void **) &d_csrValL, nnnz * sizeof(double));
//
//    cudaMemcpy(d_csrRowPtrL, rowPtr, (nrow + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrColIdx, colInd, nnnz * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrValL, csrData, nnnz * sizeof(double), cudaMemcpyHostToDevice);
//
//    // Vector b
//    cudaMalloc((void **) &d_b, nrow * sizeof(double));
//    cudaMemcpy(d_b, b, nrow * sizeof(double), cudaMemcpyHostToDevice);
//
//    cudaMalloc((void **) &d_x, nrow * sizeof(double));
//    cudaMemcpy(d_x, x, nrow * sizeof(double), cudaMemcpyHostToDevice);
//    spTrSolve(d_csrRowPtrL, d_csrColIdx, d_csrValL, nrow, nnnz, d_b, d_x, false);
//
//    cudaMemcpy(x, d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nrow; i++) {
//        std::cout << std::setprecision(12) << x[i] << "\n";
//    }
//
//}

//int main(int argc, char *argv[]) {
//    std::ifstream in1;
//    in1.open("/home/featurize/data/mock.mtx");
//    int tNrow = 351864;
//    int *tRowPtr = new int[tNrow + 1]{};
//    for (int i = 0; i < tNrow + 1; ++i) {
//        in1 >> tRowPtr[i];
//    }
//    int tNnnz = tRowPtr[tNrow];
//    int *tColInd = new int[tNnnz]{};
//    double *tCsrData = new double[tNnnz]{};
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tColInd[i];
//    }
//    for (int i = 0; i < tNnnz; ++i) {
//        in1 >> tCsrData[i];
//    }
//    double *tB = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        in1 >> tB[i];
//    }
//    double *x = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        x[i] = 0;
//    }
//    double *rx = new double[tNrow];
//    for (int i = 0; i < tNrow; ++i) {
//        in1 >> rx[i];
//    }
//    int nrow, nnnz;
//    int *rowPtr, *colInd;
//    double *csrData, *b;
//    int expectRow = 100000;
//    if (expectRow != -1) {
//        nrow = expectRow;
//        rowPtr = new int[nrow + 1];
//        for (int i = 0; i < nrow + 1; ++i) {
//            rowPtr[i] = tRowPtr[i];
//        }
//        nnnz = tRowPtr[nrow];
//        colInd = new int[nnnz];
//        csrData = new double[nnnz];
//        for (int i = 0; i < nnnz; ++i) {
//            colInd[i] = tColInd[i];
//        }
//        for (int i = 0; i < nnnz; ++i) {
//            csrData[i] = tCsrData[i];
//        }
//        b = new double[nrow];
//        for (int i = 0; i < nrow; ++i) {
//            b[i] = tB[i];
//        }
//    }
//
//    int *d_csrRowPtrL;
//    int *d_csrColIdx;
//    int *d_get_value;
//    double *d_csrValL;
//    double *d_b;
//    double *d_x;
//
//
//    // Matrix L
//    cudaMalloc((void **) &d_csrRowPtrL, (nrow + 1) * sizeof(int));
//    cudaMalloc((void **) &d_csrColIdx, nnnz * sizeof(int));
//    cudaMalloc((void **) &d_csrValL, nnnz * sizeof(double));
//
//    cudaMemcpy(d_csrRowPtrL, rowPtr, (nrow + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrColIdx, colInd, nnnz * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_csrValL, csrData, nnnz * sizeof(double), cudaMemcpyHostToDevice);
//
//    // Vector b
//    cudaMalloc((void **) &d_b, nrow * sizeof(double));
//    cudaMemcpy(d_b, b, nrow * sizeof(double), cudaMemcpyHostToDevice);
//
//    cudaMalloc((void **) &d_x, nrow * sizeof(double));
//    cudaMemcpy(d_x, x, nrow * sizeof(double), cudaMemcpyHostToDevice);
//    spTrSolve(d_csrRowPtrL, d_csrColIdx, d_csrValL, nrow, nnnz, d_b, d_x);
//
//    cudaMemcpy(x, d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost);
//    for (int i = 0; i < nrow; i++) {
//        std::cout << std::setprecision(8) <<x[i] << " ";
//    }
//
//}

int main(int argc, char *argv[]) {
    // kvlcc2 351864 351864 2426070 /home/featurize/data/kvlcc2/matrix_u_101.dat /home/featurize/data/kvlcc2/vector_rhs_u_101.dat /home/featurize/data/kvlcc2/vector_init_u_101.dat
    // suboff 1277000 1277000 8946506 /home/featurize/data/suboff/suboff_matrix_u_101.dat /home/featurize/data/suboff/suboff_vector_rhs_u_101.dat /home/featurize/data/suboff/suboff_vector_init_u_101.dat
    int nrow = std::stoi(argv[1]);
    int ncol = std::stoi(argv[2]);
    int nnnz = std::stoi(argv[3]);
    char *matrixfile = argv[4];
    char *rhsfile = argv[5];
    char *initfile = argv[6];
    char *resultfile = argv[7];
    std::ifstream in1;
    in1.open(matrixfile);

    int k;
    int colt;
    double val;

    int *rowPtr = new int[nrow + 1]();
    int *colInd = new int[nnnz];
    int *rowMap = new int[nnnz];
    int *colSortMap = new int[nnnz];
    double *csrData = new double[nnnz];
    double *b = new double[nrow];
    double *x = new double[nrow];
    time_t run_start1 = clock();
//        printArrIC(rowPtr);
    for (int i = 0; i < nnnz; i++) {
        in1 >> k;
        in1 >> colt;
        rowPtr[k]++;
        colInd[i] = colt - 1;
        in1 >> std::scientific >> csrData[i];
        rowMap[i] = k - 1;
        if (i == 0 || k - 1 != rowMap[i - 1]) {
            colSortMap[i] = 0;
        } else {
            colSortMap[i] = colSortMap[i - 1] + 1;
        }
    }

    in1.close();
    in1.open(rhsfile);
    std::string line;
    std::getline(in1, line);
    for (int i = 0; i < nrow; i++) {
        in1 >> std::scientific >> b[i];
        rowPtr[i + 1] += rowPtr[i];
    }
    in1.close();
    in1.open(initfile);
    std::getline(in1, line);
    time_t run_start2 = clock();
    for (int i = 0; i < nrow; i++) {
        in1 >> std::scientific >> x[i];
    }
    in1.close();
    std::ofstream out2;
    out2.open(resultfile);
    for (int i = 0; i < nrow + 1; i++) {
        out2 << rowPtr[i] << " ";
    }
    out2 << "\n";
    for (int i = 0; i < nnnz; i++) {
        out2 << colInd[i] << " ";
    }
    out2 << "\n";
    for (int i = 0; i < nnnz; i++) {
        out2 << csrData[i] << " ";
    }
    out2.flush();



    solveMat(nrow, nnnz, rowPtr, colInd, csrData, rowMap, colSortMap, b, x);
    std::ofstream out1;
    out1.open(resultfile);
    out1 << "Vec_s = [\n";
    for (int i = 0; i < nrow; i++) {
        out1 << std::setprecision(18) << x[i] << "\n";
    }
    out1 << "]\n";

}