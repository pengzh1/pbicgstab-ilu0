export MIMALLOC_SHOW_STATS=1
export LD_PRELOAD=/usr/local/lib/libmimalloc.so
mpirun -np 1 ./megasolve dboat /root/data/dboat 0
mpirun -np 1 ./megasolve dboat /root/data/dboat 1
mpirun -np 1 ./megasolve dboat /root/data/dboat 2
mpirun -np 1 ./megasolve dboat /root/data/dboat 3
mpirun -np 1 ./megasolve dboat /root/data/dboat 4
mpirun -np 1 ./megasolve dboat /root/data/dboat 5
mpirun -np 1 ./megasolve dboat /root/data/dboat 6 12