#!/bin/bash

g++ -std=c++17 -o test_base test_base.cpp src/base/base.cpp src/base/alloc.cpp -I./include -lgtest -lgtest_main -pthread -lglog
./test_base

