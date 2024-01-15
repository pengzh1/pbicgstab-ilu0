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
#include "solve.hpp"


int main(int argc, char *argv[]) {
    std::cout << "使用方式及参数顺序: ./megasolve [矩阵行数] [非零元数] [矩阵文件] [右端项文件] [初始值文件] [输出结果文件] \n";
    // kvlcc2 351864 2426070 /home/featurize/data/kvlcc2/matrix_u_101.dat /home/featurize/data/kvlcc2/vector_rhs_u_101.dat /home/featurize/data/kvlcc2/vector_init_u_101.dat
    // suboff 1277000 8946506 /home/featurize/data/suboff/suboff_matrix_u_101.dat /home/featurize/data/suboff/suboff_vector_rhs_u_101.dat /home/featurize/data/suboff/suboff_vector_init_u_101.dat
    int nrow = std::stoi(argv[1]);
    int nnnz = std::stoi(argv[2]);
    char *matrixfile = argv[3];
    char *rhsfile = argv[4];
    char *initfile = argv[5];
    char *resultfile = argv[6];
    std::ifstream in1;
    gettm(getTimeStamp());
    std::cout <<"-----1.开始读取文件\n";
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
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    cusparseCreate(&cusparseHandle);
    cublasCreate_v2(&cublasHandle);
    time_t run_start = clock();
    int n = nrow;
    int nnz = nnnz;
    time_t solve_start = clock();
    solve(rowPtr, colInd, csrData, rowMap, colSortMap, b, x, n, nnz, tol, cusparseHandle, cublasHandle);
    time_t solve_end = clock();
    for (int i = 0; i < 5; ++i) {
        std::cout << std::setprecision(12) << x[i] << " ";
    }
    std::cout << std::endl;
    cusparseDestroy(cusparseHandle);
    cublasDestroy_v2(cublasHandle);
    std::ofstream out1;
    out1.open(resultfile);
    out1 << "Vec_s = [\n";
    for (int i = 0; i < nrow; i++) {
        out1 << std::setprecision(18) << x[i] << "\n";
    }
    out1 << "]\n";

}