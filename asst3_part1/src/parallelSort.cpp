/* Copyright 2014 15418 Staff */

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>

#include "parallelSort.h"
#include "CycleTimer.h"
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
  srand(time(0));
    for(size_t i = 0; i < sampleSize; i++)
    {
       sample[i] = data[rand()%dataSize]; 
    }
}

int partition(float *data, int left, int right, int pivotIndex)
{
    float pivotValue = data[pivotIndex];
    float tmp = data[right];
    data[right] = data[pivotIndex];
    data[pivotIndex] = tmp;
    int storeIndex = left;
    for(int i = left; i < right; i++)
    {
        if(data[i] <= pivotValue)
        {       
            tmp = data[storeIndex];
            data[storeIndex] = data[i];
            data[i] = tmp;
            storeIndex++; 
        }
    }  
    
    tmp = data[right];
    data[right] = data[storeIndex];
    data[storeIndex] = tmp;
       

    return storeIndex; 
}

void quicksort(float *data, int left, int right)
{
    if(left < right)
    {
        int pivotIndex = left + (right-left)/2;
        //choose pivot
        int pivotNewIndex = partition(data, left, right, pivotIndex);
        quicksort(data, left, pivotNewIndex - 1);
        quicksort(data, pivotNewIndex + 1, right);
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

    double startProcessorTime = CycleTimer::currentSeconds();

    //Each processor sends S samples to processor 0 
    int S = (int)log(dataSize)*12;
    if(S > localSize)
        S = (localSize + 1)/ 2;
    int pivotNum = S / procs;
    int * bucketPivot = (int *)malloc(sizeof(int)*procs);


    float * pivots = (float *) malloc((procs - 1) * sizeof(float));
    int totalSamples = S*procs;
    if(procId == 0)
    {
        float * samples = (float *)malloc(totalSamples*sizeof(float));
        randomSample(data, localSize, samples, S);

        for(int i = 1; i < procs; i++)
        {
            MPI_Recv(&samples[i*S], S, MPI_FLOAT, i, 201, MPI_COMM_WORLD, NULL);
        }
        for(int i = 0; i < procs-1; i++)
        {       
            pivots[i] = samples[(i+1) * totalSamples / procs];
        }   
        
        quicksort(pivots, 0, procs - 2);
        float * temp;


        for(int i = 1; i < procs; i++)
        {
            MPI_Send(pivots, procs - 1, MPI_FLOAT, i, 201, MPI_COMM_WORLD);
        }   

    }
    else
    {
        float * samples = (float *)malloc(totalSamples*sizeof(float));
        randomSample(data, localSize, samples, S);   
        MPI_Send(samples, S, MPI_FLOAT, 0, 201, MPI_COMM_WORLD);
        MPI_Recv(pivots, procs - 1, MPI_FLOAT, 0, 201, MPI_COMM_WORLD, NULL);
        
    }

   
    //count number of elements in each bucket
    int * countBucket = (int *)malloc(sizeof(int)*procs); 
    //used for inserting elements in specific buckets
    int * counter = (int *)malloc(sizeof(int)*procs);
    int * getCount = (int *) malloc(sizeof(int) * procs);
    int * localDisplacement = (int *)malloc(sizeof(int) * procs);
    for(int i = 0; i < procs; i++)
    {
        countBucket[i] = 0;
        counter[i] = 0;
        getCount[i] = 0;
        localDisplacement[i] = 0;
    }  

    for(int i = 0; i < localSize; i++)
    {  
        int x = upper_bound(pivots, pivots + procs - 1, data[i]) - pivots; 
        countBucket[x] = countBucket[x]+1; 
    }

    float * buckets = (float *)malloc(sizeof(float) * localSize);
    int * displacement = (int *)malloc(sizeof(int)*procs);
    for(int i = 0; i < procs; i++)
    {
        if(i == 0)
            displacement[i] = 0;
        else
            displacement[i] = countBucket[i-1] + displacement[i-1];
    }
    for(int i = 0; i < localSize; i++)
    {
        int x = upper_bound(pivots,pivots + procs - 1, data[i]) - pivots;
        buckets[counter[x] + displacement[x]] = data[i];
        counter[x] = counter[x] + 1; 
    }

    MPI_Alltoall(countBucket, 1, MPI_INT, getCount, 1, MPI_INT, MPI_COMM_WORLD);


     
    int sumTotal = 0;
    for (int i = 0; i < procs;i++)
    {
        sumTotal += getCount[i];
        if(i == 0)
            localDisplacement[0] = 0;
        else
            localDisplacement[i] = localDisplacement[i - 1] + getCount[i - 1]; 

    }
    
    float * finalSort = (float *)malloc(sizeof(float)*sumTotal);
    MPI_Alltoallv(buckets, countBucket, displacement, MPI_FLOAT, finalSort, getCount, localDisplacement, MPI_FLOAT, MPI_COMM_WORLD);
    sort(finalSort, finalSort + sumTotal);
    localSize = sumTotal;
    sortedData = finalSort;

   return;

}


void localSort(float *data, size_t dataSize)
{

    if(dataSize <= 1)
        return;
    int random = rand() * dataSize;


}
