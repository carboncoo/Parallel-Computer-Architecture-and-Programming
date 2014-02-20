#ifndef _STLSORT_H_
#define _STLSORT_H_

//Single processor serial sort
double serialSort(float *data, int procs, int procId, size_t dataSize, size_t localSize);

//Parallel merge sort
double mergeSort(float *data, int procs, int procId, size_t dataSize, size_t localSize);

#endif
