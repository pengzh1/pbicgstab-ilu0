#ifndef AMG_ILU_CUH
#define AMG_ILU_CUH

#endif //AMG_ILU_CUH

void ILU0_MEGA(const int *d_csrRowPtr,
               const int *d_csrColIdx,
               double *d_csrVal,
               const int m, // rows
                          const int nnnz,const int dep_size,const int dep_sub_size);

void find_locn_of_diag_elements(const int nrows, int *const __restrict__ diag_ptrs,
                                const int *const __restrict__ row_ptrs,
                                const int *const __restrict__ col_idxs);
