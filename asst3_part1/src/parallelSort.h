#ifndef _PARALLELSORT_H_
#define _PARALLELSORT_H_

//#define NO_DEBUG

#define ROOT 0

//MPI implementation of parallel sort here
void parallelSort(float *data, float *&sortedData, int procs, int procId, size_t dataSize, size_t &localSize);

#endif
