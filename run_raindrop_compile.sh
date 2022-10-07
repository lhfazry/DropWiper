#!/bin/bash
cd drop_generator/makeRain/ 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

cd ../../rainEdge
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8