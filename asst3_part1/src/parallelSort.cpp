/* Copyright 2014 15418 Staff */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>

#include "parallelSort.h"

using namespace std;

void printArr(const char* arrName, int *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %d %d %d %d\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void printArr(const char* arrName, float *arr, size_t size, int procId) {
#ifndef NO_DEBUG
  for(size_t i=0; i<size; i+=4) {
    printf("%s[%d:%d] on processor %d = %f %f %f %f\n", arrName, i,
        min(i+3,size-1), procId, arr[i], (i+1 < size) ? arr[i+1] : 0, 
        (i+2 < size) ? arr[i+2] : 0, (i+3 < size) ? arr[i+3] : 0); 
  }
#endif
}

void randomSample(float *data, size_t dataSize, float *sample, size_t sampleSize) {
  for (size_t i=0; i<sampleSize; i++) {
    sample[i] = data[rand()%dataSize];
  }
}

void parallelSort(float *data, float *&sortedData, int procs, int procId, size_t dataSize, size_t &localSize) {
  // Implement parallel sort algorithm as described in assignment 3
  // handout. 
  // Input:
  //  data[]: input arrays of unsorted data, distributed across p processors
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  //
  // Step 1: Choosing Pivots to Define Buckets
  // Step 2: Bucketing Elements of the Input Array
  // Step 3: Redistributing Elements
  // Step 5: Final Local Sort
  // ***********************************************************************








  // Output:
  //  sortedData[]: output arrays of sorted data, initially unallocated
  //                please update the size of sortedData[] to localSize!
  localSize = 0;
  return;
}

