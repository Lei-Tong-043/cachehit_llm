#!/bin/bash

cd ./build
cmake ..
make -j4
cd test

./test_llm
