/* Copyright 2014 15418 Staff */

#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <mpi.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <unistd.h>  // For STDIN_FILENO.

#include "stlSort.h"
#include "parallelSort.h"
#include "dataGen.h"

using namespace std;

#define SEED 0

enum Distribution {
  NORMAL,
  EXPONENTIAL,
  BAD1
};

void usage(char *program);
void parse_args(int argc, char **argv, float *&data, int procs, int procId, size_t &dataSize, size_t &localSize);
void parse_input(float *&data, int procs, int procId, size_t &dataSize, size_t &localSize);
inline void allocData(float *&data, int procs, int procId, size_t &dataSize, size_t &localSize);
void checkSort(float *sortedData, int procs, int procId, size_t dataSize, size_t localSize);

int main(int argc, char **argv) {

  //The total number of processes in use (P)
  int procs;

  //This processor id. (i)
  int procId;

  //Dataset size to sort (N)
  size_t dataSize;

  //Size of dataset on this processes (N/P)
  size_t localSize;

  MPI_Init(&argc, &argv);

  /* Find out how many processes we are using and which one we are. */
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procId);

  srand(SEED+procId);
  float *data;        // Dataset to sort
  float *sortedData;  // Sorted dataset

  parse_args(argc, argv, data, procs, procId, dataSize, localSize);
  assert(dataSize > procs);

  double serialTime = serialSort(data, procs, procId, dataSize, localSize);
  double mergeTime = mergeSort(data, procs, procId, dataSize, localSize);

  double startTime = MPI_Wtime();

  sortedData = NULL;
  parallelSort(data, sortedData, procs, procId, dataSize, localSize);

  double endTime = MPI_Wtime();

  checkSort(sortedData, procs, procId, dataSize, localSize);
  if (procId == 0) {
    printf("Serial sort\t\ttook %.4fs on %d processors\n",
        serialTime, 1);
    printf("Parallel merge sort\ttook %.4fs on %d processors\tSpeedup: %.4fx\n",
        mergeTime, procs, serialTime/mergeTime);
    printf("Solution\t\ttook %.4fs on %d processors\tSpeedup: %.4fx\n",
        endTime - startTime, procs, serialTime/(endTime - startTime));
  }

  MPI_Finalize();
  if (data) free(data);
  if (localSize && sortedData) free(sortedData);
  return 0;
}

void usage(char *program) {
  printf("Usage: %s [options]\n"
      "  Sort a random or the input dataset \n"
      "\n"
      "Program Options:\n"
      "  -s  --size <N>           Size of dataset to sort\n"
      "  -d  --dist exp|norm|bad1 Distribution to generate dataset\n"
      "  -p  --par <pram>         Use <pram> as distribution parameter\n"
      "  -a  --almost <swap>      use <swap> comparisons to generate almost sorted dataset\n"
      "  -i  --input <file>       Use <file> instead of generated dataset\n"
      "  -?  --help               This message\n", program);
}

void parse_args(int argc, char **argv, float *&data, int procs, int procId, size_t &dataSize, size_t &localSize) {
  int opt;
  Distribution distribution;
  float parameter;
  bool input;
  size_t almostSorted;

  //Default
  dataSize = 1000;
  distribution = NORMAL;
  parameter = 5.f;
  input = false;
  almostSorted = 0;

  static struct option long_opts[] = {
    {"size", 1, 0, 's'},
    {"dist", 1, 0, 'd'},
    {"par", 1, 0, 'p'},
    {"almost", 1, 0, 'a'},
    {"input", 1, 0, 'i'},
    {"help", 0, 0, '?'},
    {0, 0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "s:d:p:a:i:?h", long_opts, NULL)) != EOF) {
    switch (opt) {
      case 's':
        dataSize = atoi(optarg);
        break;
      case 'd':
        if (strcmp(optarg, "exp") == 0) {
          distribution = EXPONENTIAL;
        } else if (strcmp(optarg, "norm") == 0) {
          distribution = NORMAL;
        } else if (strcmp(optarg, "bad1") == 0) {
          distribution = BAD1;
        } else {
          perror(optarg);
        }
        break;
      case 'p':
        parameter = atof(optarg);
        if (parameter == 0.f) {
          perror(optarg);
        }
        break;
      case 'a':
        almostSorted = atoi(optarg);
        break;
      case 'i':
        if (dup2(open(optarg, O_RDONLY), STDIN_FILENO) < 0) {
          perror(optarg);
          exit(EXIT_FAILURE);
        }
        input = true;
        break;
      case 'h':                  /* Explicit fall through */
      case '?':
        usage(argv[0]);
        exit(EXIT_SUCCESS);
      default:
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  if (input == true) {
    parse_input(data, procs, procId, dataSize, localSize);
  } else if (input == false) {
    allocData(data, procs, procId, dataSize, localSize);
    if (distribution == NORMAL) {
      uniform(data, localSize, 0.f, parameter, procId);
    } else if (distribution == EXPONENTIAL) {
      exponential(data, localSize, parameter, procId);
    } else if (distribution == BAD1) {
      float interval = parameter/procs;
      uniform(data, localSize, procId * interval, (procId+1) * interval, procId);
    }
  }
  if (almostSorted) {
    almostSort(data, localSize, almostSorted);
  }
}

void parse_input(float *&data, int procs, int procId, size_t &dataSize, size_t &localSize) {
  scanf("%u", &dataSize);
  allocData(data, procs, procId, dataSize, localSize);

  //TODO: each processor grab its own data
  for (size_t i=0; i<dataSize; i++) {
    scanf("%f", &data[i]);
  }
  printf("Processor %d finished loading...\n", procId);
}

inline void allocData(float *&data, int procs, int procId, size_t &dataSize, size_t &localSize) {
  localSize = (procId < (dataSize%procs)) ? 
    (dataSize/procs + 1) : dataSize/procs;
  data = (float *)malloc(sizeof(float) * localSize);
}

void checkSort(float *sortedData, int procs, int procId, size_t dataSize, size_t localSize) {
  if (!localSize) {
    printf("@@@ Skipping check results for processor %d because of zero localSize!\n", procId);
    return;
  }
  if (!sortedData) {
    printf("@@@ Wrong Result! NULL pointer returned for sortedData[]!\n");
    exit(EXIT_SUCCESS);
  }
  for (size_t i=1; i<localSize; i++) {
    if (sortedData[i-1] > sortedData[i]) {
      printf("@@@ Wrong Result @ sortedData[%d:%d] = %f %f!\n", 
          i-1, i, sortedData[i-1], sortedData[i]);
      exit(EXIT_SUCCESS);
    }
  }
  float *bucketStart;
  float *bucketEnd;
  size_t bucketSize;
  if (procId == ROOT) {
    bucketStart = (float *)malloc(sizeof(float) * procs);
    bucketEnd = (float *)malloc(sizeof(float) * procs);
  }
  assert( MPI_Gather(sortedData, 1, MPI_FLOAT,
        bucketStart, 1, MPI_FLOAT,
        ROOT, MPI_COMM_WORLD) == MPI_SUCCESS );
  assert( MPI_Gather(sortedData + localSize - 1, 1, MPI_FLOAT,
        bucketEnd, 1, MPI_FLOAT,
        ROOT, MPI_COMM_WORLD) == MPI_SUCCESS );
  assert( MPI_Reduce(&localSize, &bucketSize, 1, MPI_INT,//MPI_UNSIGNED_LONG,
        MPI_SUM, ROOT, MPI_COMM_WORLD) == MPI_SUCCESS );
  if (procId == ROOT) {
    if ((unsigned int)dataSize != (unsigned int)bucketSize) {
      printf("@@@ Wrong Result! dataSize %lld does not match with total sortedData size %lld!\n", dataSize, bucketSize);
      exit(EXIT_SUCCESS);
    }
    for (size_t i=1; i<procs; i++) {
      if (bucketEnd[i-1] > bucketStart[i]) {
        printf("@@@ Wrong Result! Bucket %d (%f~%f) and %d (%f~%f) overlap!\n", i-1, bucketStart[i-1], bucketEnd[i-1], i, bucketStart[i], bucketEnd[i]);
        exit(EXIT_SUCCESS);
      }
    }
    printf("Result validation for %d numbers passed!\n", dataSize);
  }
}

