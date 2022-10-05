#!/bin/bash

cmake -DCMAKE_PREFIX_PATH=/usr/local/lib64/cmake/opencv4 ../src
cmake --build . --target all -j 8

