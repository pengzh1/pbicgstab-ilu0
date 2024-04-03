export MIMALLOC_SHOW_STATS=1
export LD_PRELOAD=/usr/local/lib/libmimalloc.so
mpirun -np 1 ./megasolve kcs /root/data/kcs 0
mpirun -np 1 ./megasolve kcs /root/data/kcs 1
mpirun -np 1 ./megasolve kcs /root/data/kcs 2
mpirun -np 1 ./megasolve kcs /root/data/kcs 3
mpirun -np 1 ./megasolve kcs /root/data/kcs 4
mpirun -np 1 ./megasolve kcs /root/data/kcs 5
mpirun -np 1 ./megasolve kcs /root/data/kcs 6