# STEREYE CUDA TUTORIAL

1. Testing Environment
    - Ubuntu 18.04
    - CUDA 11.3
    - RTX 3060

2. Build
    -   `mkdir build && cd build && cmake .. && make`

3. Topics
    - `example_01`: query device info; build with CMake;
    - `example_02`: loop parallelism;  shared library compilation;
    - `example_03`: cuda timing; thrust introduction; grid stride loop;
    - `example_04`: compile with openmp ;locality principle(shared memory utilization);
    - `example_05`: atomic operation;
    - `example_06`: parallel reduction;
    - `example_07`: dynamic shared memory; unified memory; asynchronous stream;