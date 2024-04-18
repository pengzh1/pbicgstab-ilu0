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
#include <unistd.h>
#include "string.h"
#include <mpi.h>

#ifndef AMG_SOLVE_HPP
#define AMG_SOLVE_HPP

#endif //AMG_SOLVE_HPP
using namespace std;


bool new_data = true;
bool mdebug = false;
double tol = 1e-6;

struct solveMeta {
    int mid;
    string task_name;
    string result_file;
    int *rowPtr;
    int *colInd;
    double *csrData;
    double *b;
    double *x;
    int n;
    int nnz;
    double tol;
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    int dep_size;
    int dep_sub_size;
};

//string kcs[7][7] = {
//        {
//                "589940", "4141924", "matrix_u_101.dat",
//                "vector_rhs_u_101.dat", "vector_init_u_101.dat", "result_u.dat", "动量分量-u"
//        },
//        {
//                "589940", "4141924", "matrix_v_101.dat",
//                "vector_rhs_v_101.dat", "vector_init_v_101.dat", "result_v.dat", "动量分量-v"
//        },
//        {
//                "589940", "4141924", "matrix_w_101.dat",
//                "vector_rhs_w_101.dat", "vector_init_w_101.dat", "result_w.dat", "动量分量-w"
//        },
//        {
//                "589940", "4141924", "matrix_s_101.dat",
//                "vector_rhs_s_101.dat", "vector_init_s_101.dat", "result_s.dat", "流体体积分数"
//        },
//        {
//                "589940", "4049160", "matrix_e_101.dat",
//                "vector_rhs_e_101.dat", "vector_init_e_101.dat", "result_e.dat", "湍流耗散率"
//        },
//        {
//                "589940", "4141924", "matrix_k_101.dat",
//                "vector_rhs_k_101.dat", "vector_init_k_101.dat", "result_k.dat", "湍动能"
//        },
//        {
//                "589940", "4141924", "matrix_p_101.dat",
//                "vector_rhs_p_101.dat", "vector_init_p_101.dat", "result_p.dat", "压力修正值"
//        }
//};

string dboat[7][7] = {
        {
                "2849291", "20149067", "matrix_u_101.dat",
                "vector_rhs_u_101.dat", "vector_init_u_101.dat", "result_u.dat", "动量分量-u"
        },
        {
                "2849291", "20149067", "matrix_v_101.dat",
                "vector_rhs_v_101.dat", "vector_init_v_101.dat", "result_v.dat", "动量分量-v"
        },
        {
                "2849291", "20149067", "matrix_w_101.dat",
                "vector_rhs_w_101.dat", "vector_init_w_101.dat", "result_w.dat", "动量分量-w"
        },
        {
                "2849291", "20149067", "matrix_s_101.dat",
                "vector_rhs_s_101.dat", "vector_init_s_101.dat", "result_u.dat", "流体体积分数"
        },
        {
                "2849291", "19357108", "matrix_e_101.dat",
                "vector_rhs_e_101.dat", "vector_init_e_101.dat", "result_e.dat", "湍流耗散率"
        },
        {
                "2849291", "20149067", "matrix_k_101.dat",
                "vector_rhs_k_101.dat", "vector_init_k_101.dat", "result_k.dat", "湍动能"
        },
        {
                "2849291", "20149067", "matrix_p_101.dat",
                "vector_rhs_p_101.dat", "vector_init_p_101.dat", "result_p.dat", "压力修正值"
        },
};

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

void printTime() {
    // 获取当前时间（自epoch以来的时间）
    auto now = std::chrono::system_clock::now();

    // 转换为自epoch以来的微秒数
    auto duration = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    // 分离出秒和微秒部分
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(microseconds);
    auto micros = microseconds.count() % static_cast<std::chrono::microseconds::rep>(1000000);

    // 将秒转换为time_t以便使用标准C时间函数
    std::time_t time = std::chrono::system_clock::to_time_t(now - (microseconds - seconds));

    // 转换为结构体tm以便使用printf格式化
    std::tm *tm = std::localtime(&time);

    // 使用printf打印时间戳
    printf("%04d-%02d-%02d %02d:%02d:%02d.%06ld ",
           tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
           tm->tm_hour, tm->tm_min, tm->tm_sec,
           micros);
}

void cudaCheckError() {
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "Cuda failure: '" << cudaGetErrorString(e) << "'";
        throw "Cuda failure";
    }
}

void solve(int mpid, int *rowPtr, int *colInd, double *csrData,
           double *b, double *x, int n, int nnz, double tol, cusparseHandle_t cusparseHandle,
           cublasHandle_t cublasHandle, int dep_size, int dep_sub_size,int max_iter) {
    time_t solve_start = clock();
    int *gColInd, *gRowPtr;
    double *gCsrData, *gB, *gX;
    // 分配内存，
    // gCOlInd csr格式矩阵非零元素列指针
    // gRowPtr csr格式矩阵非零元素行指针
    // gCsrData csr格式矩阵非零元素值指针
    // gB 右端项
    // gX 初始值
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
//    cudaCheckError();
    // Solve Ax = b for x.
    spSolverBiCGStab(mpid, n, nnz, gCsrData, gRowPtr, gColInd, gB, gX, tol, cusparseHandle,
                     cublasHandle, dep_size, dep_sub_size, max_iter);
    time_t stab_end = clock();
    // Copy x back to CPU.
    cudaMemcpy(x, gX, n * sizeof(double), cudaMemcpyDeviceToHost);
    time_t solve_end = clock();
}

void printProgressBar(int progress, int total) {
    int barWidth = 50;
    float percentage = (float) progress / total;
    int filledLength = barWidth * percentage;
    // 打印文件读取进度条
    printf("\r[");
    for (int i = 0; i < barWidth; i++) {
        if (i < filledLength) {
            printf("#");
        } else {
            printf("-");
        }
    }
    printf("] %.1f%%", percentage * 100);
    // 输出
    fflush(stdout);
}

char *readAll(string filename) {
    FILE *file = fopen(filename.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    char *buffer = (char *) malloc(file_size + 1);
    size_t bytes_read = fread(buffer, 1, file_size, file);
    buffer[bytes_read] = '\0';
    fclose(file);
    return buffer;
}

solveMeta *read_new(string matrix_file, string rhs_file, string init_file,
                    string result_file, int nrow,
                    int nnnz, int dep_size, int dep_sub_size) {

    printTime();
    printf("---开始读取数据 \n");
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    cusparseCreate(&cusparseHandle);
    cublasCreate_v2(&cublasHandle);
    double val;
    int *rowPtr, *colInd;
    double *csrData, *b, *x;
    cudaMallocHost((void **) &rowPtr, (nrow + 1) * sizeof(int));
    cudaMallocHost((void **) &colInd, (nnnz) * sizeof(int));
    cudaMallocHost((void **) &csrData, (nnnz) * sizeof(double));
    cudaMallocHost((void **) &b, (nrow) * sizeof(double));
    cudaMallocHost((void **) &x, (nrow) * sizeof(double));
    char *buffer = readAll(matrix_file);
    int maxRow = 0;
    double c = 0.0;
    char *end;
    char *linestr = buffer;
    int k;
    int colt;
    int rnnz = 0;
    for (int j = 0; j < nnnz; ++j) {
        if (j % 100000 == 0) {
            printProgressBar(j, nnnz + 2 * nrow);
        }
        k = strtol(linestr, &end, 10);
        linestr = end + 1;
        colt = strtol(linestr, &end, 10);
        linestr = end + 1;
        c = strtod(linestr, &end);
        linestr = end + 1;
        if (rowPtr[k] != 0 && colt == colInd[rnnz - 1] + 1) {
            csrData[rnnz - 1] += c;
        } else {
            rowPtr[k]++;
            if (rowPtr[k] > maxRow) {
                maxRow = rowPtr[k];
            }
            colInd[rnnz] = colt - 1;
            csrData[rnnz] = c;
            rnnz++;
        }
    }
    // 释放内存
    free(buffer);
    buffer = readAll(rhs_file);
    linestr = buffer + 9;
    for (int j = 0; j < nrow; j++) {
        if (j % 100000 == 0) {
            printProgressBar(j + nnnz, nnnz + 2 * nrow);
        }
        c = strtod(linestr, &end);
        linestr = end + 1;
        b[j] = c;
        rowPtr[j + 1] += rowPtr[j];
    }
    // 释放内存
    free(buffer);
    buffer = readAll(init_file);
    linestr = buffer + 9;
    if (new_data) {
        for (int j = 0; j < nrow; j++) {
            if (j % 100000 == 0) {
                printProgressBar(j + nnnz + nrow, nnnz + 2 * nrow);
            }
            linestr = strchr(linestr, ' ') + 1;
            linestr = strchr(linestr, ' ') + 1;
            c = strtod(linestr, &end);
            linestr = strchr(linestr, '\n');
            x[j] = c;
        }
    } else {
        for (int j = 0; j < nrow; j++) {
            if (j % 100000 == 0) {
                printProgressBar(j + nnnz + nrow, nnnz + 2 * nrow);
            }
            c = strtod(linestr, &end);
            linestr = end + 1;
            x[j] = c;
        }
    }
    // 释放内存
    free(buffer);
    printProgressBar(nrow + nnnz + nrow, nnnz + 2 * nrow);
    printf("\n");
    printTime();
    printf("---数据读取完成 \n");
    solveMeta *meta = new solveMeta;
    meta->result_file = result_file;
    meta->rowPtr = rowPtr;
    meta->colInd = colInd;
    meta->csrData = csrData;
    meta->b = b;
    meta->x = x;
    meta->n = nrow;
    meta->nnz = rnnz;
    meta->tol = tol;
    meta->cusparseHandle = cusparseHandle;
    meta->cublasHandle = cublasHandle;
    meta->dep_size = dep_size;
    meta->dep_sub_size = dep_sub_size;
    return meta;
}


void solve_new(int mid, string task_name, string result_file, int *rowPtr, int *colInd, double *csrData,
               double *b, double *x, int n, int nnz, double tol, cusparseHandle_t cusparseHandle,
               cublasHandle_t cublasHandle, int dep_size, int dep_sub_size,int max_iter) {
    time_t solve_start = clock();
    solve(mid, rowPtr, colInd, csrData, b, x, n, nnz, tol,
          cusparseHandle, cublasHandle, dep_size, dep_sub_size,max_iter);
    time_t solve_end = clock();
    gettm(getTimeStamp());
    printf("-----[%s]3.求解完成 耗时 %ld ms \n", task_name.c_str(),
           (solve_end - solve_start) / (CLOCKS_PER_SEC / 1000));
}

void final_new(double *x, int n, int nnz, string result_file, cusparseHandle_t cusparseHandle,
               cublasHandle_t cublasHandle) {
    //    free(csrData);
//    free(colInd);
//    free(rowPtr);
    if (mdebug) {
        for (int i = 0; i < 4; ++i) {
            cout << setprecision(8) << x[i] << " ";
        }
        cout << "...";
        for (int i = n - 4; i < n; ++i) {
            cout << setprecision(8) << x[i] << " ";
        }
        cout << endl;
    }
    cusparseDestroy(cusparseHandle);
    cublasDestroy_v2(cublasHandle);
    ofstream out1;
    out1.open(result_file);
    out1 << "Vec_s = [\n";
    for (int i = 0; i < n; i++) {
        out1 << setprecision(18) << x[i] << "\n";
    }
    out1 << "]\n";
}
