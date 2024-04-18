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
#include "string.h"
#include <unistd.h>
#include <cmath>

using namespace std;

// /root/data/kcs matrix_u_101.dat vector_rhs_u_101.dat vector_init_u_101.dat 589940 4141924 result_u.dat 6
int main(int argc, char *argv[]) {
    cout << "使用方式及参数顺序: ./megasolve [数据目录] [系数矩阵文件名] [右端项文件名] [初始值文件名] [网格数量] [非零元数]"
            " [输出结果文件名] [可选-最大迭代次数]\n";
    char *dir = argv[1];
    char *mtxFile = argv[2];
    char *rhsFile = argv[3];
    char *initFile = argv[4];
    int nrow = stoi(argv[5]);
    int nnnz = stoi(argv[6]);
    char *resultFile = argv[7];
    int max_iter = 6;
    if (argc > 8) {
        max_iter = stoi(argv[8]);
    }
    string data_dir = dir;
    //读取和解析矩阵文件、右端项文件、初始值文件
    solveMeta *m = read_new(data_dir + "/" + mtxFile,
                            data_dir + "/" + rhsFile, data_dir + "/" + initFile,
                            data_dir + "/" + resultFile, nrow,
                            nnnz, 1024, 32);
    time_t solve_all_start = clock();
    printTime();
    printf("-----1.全部数据读取完成\n");
    solve_new(0, "求解变量", m->result_file, m->rowPtr, m->colInd, m->csrData, m->b,
              m->x, m->n, m->nnz, m->tol, m->cusparseHandle, m->cublasHandle, m->dep_size, m->dep_sub_size,
              max_iter);
    time_t solve_all_end = clock();
    printTime();
    printf("-----2.全部方程求解完成 耗时 %ld ms \n",
           (solve_all_end - solve_all_start) / (CLOCKS_PER_SEC / 1000));
    final_new(m->x, m->n, m->nnz, m->result_file, m->cusparseHandle, m->cublasHandle);
}