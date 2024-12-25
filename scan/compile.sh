rm run_test
g++   -c main.cpp         -o main.o
nvcc  -c cuda_kernel.cu   -o cuda_kernel.o 
g++  main.o cuda_kernel.o -o run_test -I/usr/local/cuda-10.2/include -L/usr/local/cuda-10.2/lib64 -lcudart -lcublas