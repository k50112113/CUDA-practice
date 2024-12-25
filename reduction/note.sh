################################################## vectorize ##################################################

==40070== NVPROF is profiling process 40070, command: ./run_test 1 0

Number of threads per block: 256
Number of elements per block: 1024
Number of blocks: 1024
==40070== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sum_reduction_kernel(float*, int, int, float*)" (done)
Done    1 internal events
==40070== Profiling application: ./run_test 1 0
==40070== Profiling result:
==40070== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "Tesla V100-DGXS-32GB (0)"
    Kernel: sum_reduction_kernel(float*, int, int, float*)
          1                            gld_efficiency                              Global Memory Load Efficiency       0.00%       0.00%       0.00%
          1                  gld_requested_throughput                           Requested Global Load Throughput  0.00000B/s  0.00000B/s  0.00000B/s
          1                            gld_throughput                                     Global Load Throughput  659.84GB/s  659.84GB/s  659.84GB/s
          1                          gld_transactions                                   Global Load Transactions      131072      131072      131072
          1              gld_transactions_per_request                       Global Load Transactions Per Request   16.000000   16.000000   16.000000
          1                      global_load_requests   Total number of global load requests from Multiprocessor        8192        8192        8192
          1                           global_hit_rate                          Global Hit Rate in unified l1/tex       0.00%       0.00%       0.00%
==40070== Trace result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.28%  1.4175ms         1  1.4175ms  1.4175ms  1.4175ms  [CUDA memcpy HtoD]
                    0.41%  5.9200us         1  5.9200us  5.9200us  5.9200us  sum_reduction_kernel(float*, int, int, float*)
                    0.18%  2.5920us         1  2.5920us  2.5920us  2.5920us  [CUDA memcpy DtoH]
                    0.12%  1.7280us         1  1.7280us  1.7280us  1.7280us  [CUDA memset]
No API activities were profiled.

################################################## no vectorize ##################################################

==40136== NVPROF is profiling process 40136, command: ./run_test 0 0
Close vectrize
Number of threads per block: 256
Number of elements per block: 1024
Number of blocks: 1024
==40136== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sum_reduction_kernel_no_vectorize(float*, int, int, float*)" (done)
Done    1 internal events
==40136== Profiling application: ./run_test 0 0
==40136== Profiling result:
==40136== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "Tesla V100-DGXS-32GB (0)"
    Kernel: sum_reduction_kernel_no_vectorize(float*, int, int, float*)
          1                            gld_efficiency                              Global Memory Load Efficiency     100.00%     100.00%     100.00%
          1                  gld_requested_throughput                           Requested Global Load Throughput  472.23GB/s  472.23GB/s  472.23GB/s
          1                            gld_throughput                                     Global Load Throughput  472.23GB/s  472.23GB/s  472.23GB/s
          1                          gld_transactions                                   Global Load Transactions      131072      131072      131072
          1              gld_transactions_per_request                       Global Load Transactions Per Request    4.000000    4.000000    4.000000
          1                      global_load_requests   Total number of global load requests from Multiprocessor       32768       32768       32768
          1                           global_hit_rate                          Global Hit Rate in unified l1/tex       0.00%       0.00%       0.00%
==40136== Trace result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.10%  1.4179ms         1  1.4179ms  1.4179ms  1.4179ms  [CUDA memcpy HtoD]
                    0.58%  8.2720us         1  8.2720us  8.2720us  8.2720us  sum_reduction_kernel_no_vectorize(float*, int, int, float*)
                    0.21%  2.9440us         1  2.9440us  2.9440us  2.9440us  [CUDA memcpy DtoH]
                    0.12%  1.7280us         1  1.7280us  1.7280us  1.7280us  [CUDA memset]
No API activities were profiled.

################################################## access global memory 4 times per loop per thread ##################################################

==39965== NVPROF is profiling process 39965, command: ./run_test 1 0
Number of threads per block: 256
Number of elements per block: 1024
Number of blocks: 1024
==39965== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sum_reduction_kernel(float*, int, int, float*)" (done)
Done    2 internal events
==39965== Profiling application: ./run_test 1 0
==39965== Profiling result:
==39965== Metric result:
Invocations                               Metric Name                                         Metric Description         Min         Max         Avg
Device "Tesla V100-DGXS-32GB (0)"
    Kernel: sum_reduction_kernel(float*, int, int, float*)
          1                            gld_efficiency                              Global Memory Load Efficiency      25.00%      25.00%      25.00%
          1                  gld_requested_throughput                           Requested Global Load Throughput  518.35GB/s  518.35GB/s  518.35GB/s
          1                            gld_throughput                                     Global Load Throughput  2073.4GB/s  2073.4GB/s  2073.4GB/s
          1                          gld_transactions                                   Global Load Transactions      524288      524288      524288
          1              gld_transactions_per_request                       Global Load Transactions Per Request   16.000000   16.000000   16.000000
          1                      global_load_requests   Total number of global load requests from Multiprocessor       32768       32768       32768
          1                           global_hit_rate                          Global Hit Rate in unified l1/tex      75.00%      75.00%      75.00%
==39965== Trace result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.18%  1.4305ms         1  1.4305ms  1.4305ms  1.4305ms  [CUDA memcpy HtoD]
                    0.52%  7.5360us         1  7.5360us  7.5360us  7.5360us  sum_reduction_kernel(float*, int, int, float*)
                    0.20%  2.9440us         1  2.9440us  2.9440us  2.9440us  [CUDA memcpy DtoH]
                    0.10%  1.3760us         1  1.3760us  1.3760us  1.3760us  [CUDA memset]
No API activities were profiled.
