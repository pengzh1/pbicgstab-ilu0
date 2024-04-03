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
#include <mpi.h>
#include <cmath>

using namespace std;

// /root/data/kvlcc2 351864 2426070
int main(int argc, char *argv[]) {
//    cout<< "使用方式及参数顺序: ./megasolve [场景名称 可选:kcs/dboat] [场景数据目录 如 /root/data/kcs] \n";
    int mid = 0, numprocs = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mid);
    char *scene = argv[1];
    char *dir = argv[2];
    int spec = -1;
    if (argc > 3) {
        spec = stoi(argv[3]);
    }
    int max_iter = 6;
    if (argc > 4) {
        max_iter = stoi(argv[4]);
    }
    string data_dir = dir;
    int num_unit = ceil(float(7) / float(numprocs));
    auto metas = new solveMeta *[num_unit];
    int nump = 0;
    for (int my_pid = mid * num_unit; my_pid < (mid + 1) * num_unit && my_pid < 7; ++my_pid) {
        if (spec > -1 && my_pid != spec) {
            continue;
        }
        //读取和解析矩阵文件、右端项文件、初始值文件
        if ("kcs" == string(scene)) {
            solveMeta *mega = read_new(my_pid, kcs[my_pid][6], data_dir + "/" + kcs[my_pid][2],
                                       data_dir + "/" + kcs[my_pid][3], data_dir + "/" + kcs[my_pid][4],
                                       data_dir + "/" + kcs[my_pid][5], stoi(kcs[my_pid][0]),
                                       stoi(kcs[my_pid][1]), 1024, 32);
            metas[nump] = mega;
        }
        if ("dboat" == string(scene)) {
            solveMeta *mega = read_new(my_pid, dboat[my_pid][6], data_dir + "/" + dboat[my_pid][2],
                                       data_dir + "/" + dboat[my_pid][3], data_dir + "/" + dboat[my_pid][4],
                                       data_dir + "/" + dboat[my_pid][5], stoi(dboat[my_pid][0]),
                                       stoi(dboat[my_pid][1]), 512, 16);
            metas[nump] = mega;
        }
        nump += 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time_t solve_all_start = clock();
    if (mid == 0) {
        printTime();
        printf("-----1.全部数据读取完成\n");
    }
    for (int i = 0; i < nump; ++i) {
        solveMeta *m = metas[i];
        solve_new(m->mid, m->task_name, m->result_file, m->rowPtr, m->colInd, m->csrData, m->b,
                  m->x, m->n, m->nnz, m->tol, m->cusparseHandle, m->cublasHandle, m->dep_size, m->dep_sub_size,
                  max_iter);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    time_t solve_all_end = clock();
    if (mid == 0) {
        printTime();
        printf("-----2.全部方程求解完成 耗时 %ld ms \n",
               (solve_all_end - solve_all_start) / (CLOCKS_PER_SEC / 1000));
    }
    for (int i = 0; i < nump; ++i) {
        solveMeta *m = metas[i];
        final_new(m->x, m->n, m->nnz, m->result_file, m->cusparseHandle, m->cublasHandle);
    }
    MPI_Finalize();
}