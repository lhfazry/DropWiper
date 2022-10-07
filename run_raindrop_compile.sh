#!/bin/bash
cd RaindropRmv/data_generation/makeRain/ 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8