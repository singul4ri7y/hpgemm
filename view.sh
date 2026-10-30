#!/usr/bin/env sh

# Single-threaded window
gnuplot -persist << 'GP'
set terminal qt 0 title "Single Thread (1T)"
set xlabel "Dimension (m = n = k)"
set ylabel "GFLOPS"
set grid
set key bottom right

plot "data/perf/mkl_1t.dat" using 1:2 with linespoints lc rgb "blue" lw 2 pt 7 title "Intel MKL", \
     "data/perf/hpgemm_1t.dat" using 1:2 with linespoints lc rgb "red" lw 2 pt 5 title "hp-gemm"
GP

# Multithreaded window
gnuplot -persist << 'GP'
set terminal qt 1 title "Multi Thread (MT)"
set xlabel "Dimension (m = n = k)"
set ylabel "GFLOPS"
set grid
set key bottom right

plot "data/perf/mkl_mt.dat" using 1:2 with linespoints lc rgb "blue" lw 2 pt 7 title "Intel MKL", \
     "data/perf/hpgemm_mt.dat" using 1:2 with linespoints lc rgb "red" lw 2 pt 5 title "hp-gemm"
GP
