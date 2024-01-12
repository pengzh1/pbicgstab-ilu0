

#include <cusparse_v2.h>
#include <cublas_v2.h>

#ifndef AMG_BICGSTAB_CUH
#define AMG_BICGSTAB_CUH

#endif //AMG_BICGSTAB_CUH


//void solve(int *rowPtr, int *colInd, double *csrData,double *b, double *x, int n, int nnz, double tol,cusparseHandle_t cusparseHandle,cublasHandle_t cublasHandle);

void spSolverBiCGStab(int n, int nnz,const  double *valA, const int *rowPtr, const int *colInd,
                      int *rowMap,int *colSortMap,
                      const double *b, double *x, double tol, cusparseHandle_t cusparseHandle,
                      cublasHandle_t cublasHandle);

int spTrSolve(const int *__restrict__ d_csrRowPtr,
              const int *__restrict__ d_csrColIdx,
              const double *__restrict__ d_csrVal,
              const int m, // rows
                               const int nnz, // nnz for L
                               const double *__restrict__ d_b, // rhs
                               double *d_x, // initVec
                               bool isL
);
