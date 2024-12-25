
# sudo /usr/local/cuda/bin/nvprof --metrics flops_sp_fma ./run_test $1
# 1,074,790,400 flops for M = K = N = 1024
sudo /usr/local/cuda/bin/nvprof --device-buffer-size 1024 --profiling-semaphore-pool-size 655360 ./run_test $1 $2
# --print-gpu-trace
# sudo /usr/local/cuda/bin/nvprof --metrics gld_efficiency --metrics global_hit_rate --metrics global_load_requests --metrics gld_requested_throughput --metrics gld_throughput ./run_test 1
# sudo /usr/local/cuda/bin/nvprof --unified-memory-profiling off --metrics flops_sp_add --metrics flops_sp_mul --metrics flops_sp_fma ./run_test 0
# /usr/local/cuda/bin/nvprof --unified-memory-profiling off ./run_test 4
# nvprof --device-buffer-size 1024 --profiling-semaphore-pool-size 655360 ./run_test