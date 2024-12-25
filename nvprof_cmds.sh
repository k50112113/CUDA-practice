sudo /usr/local/cuda/bin/nvprof \
    --metrics gld_efficiency \
    --metrics gld_requested_throughput \
    --metrics gld_throughput \
    --metrics gld_transactions \
    --metrics gld_transactions_per_request \
    --metrics gld_transactions \
    --metrics global_load_requests \
    --metrics global_hit_rate \
    --trace gpu \
    ./run_test
