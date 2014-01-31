#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#include "CycleTimer.h"

double cudaScan(int* start, int* end, int* resultarray);
double cudaScanThrust(int* start, int* end, int* resultarray);
double cudaFindRepeats(int *start, int length, int *resultarray, int *output_length);
void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --input <NAME>     Run named test. Valid tests are: random\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -t  --thrust           Use Thrust library implementation\n");
    printf("  -?  --help             This message\n");
}


/* This function contains two serial CPU implementations of exclusive scan.
 *
 * The first is *not actually parallel*, but uses a parallel algorithm - the
 * two marked for loops can be computed in parallel to speed this up.
 *
 * The second is a dead-simple serial sweep through the list.
 */
void cpu_exclusive_scan(int* start, int* end, int* output)
{
#ifdef PARALLEL
    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int twod = 1; twod < N; twod*=2)
    {
        int twod1 = twod*2;
        // parallel
        for (int i = 0; i < N; i += twod1)
        {
            output[i+twod1-1] += output[i+twod-1];
        }
    }
    output[N-1] = 0;

    // downsweep phase
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        // parallel
        for (int i = 0; i < N; i += twod1)
        {
            int tmp = output[i+twod-1];
            output[i+twod-1] = output[i+twod1-1];
            output[i+twod1-1] += tmp;
        }
    }
#endif
#ifndef PARALLEL
    int N = end - start;
    output[0] = 0;
    for (int i = 1; i < N; i++)
    {
        output[i] = output[i-1] + start[i-1];
    }
#endif
}

/* Simple serial implementation of the find_repeats function
 * Your job is to implement this computation in parallel using your parallel 
 * exclusive scan function.
 */
int cpu_find_repeats(int *start, int length, int *output){
    int count = 0, idx = 0;
    while(idx < length - 1){
        if(start[idx] == start[idx + 1]){
            output[count] = idx;
            count++;
        }
        idx++;
    }
    return count;
}

int main(int argc, char** argv)
{

    int N = 64;
    bool useThrust = false;
    std::string testName;


    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"input",      1, 0, 'i'},
        {"help",       0, 0, '?'},
        {"thrust",     0, 0, 't'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:t", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case 'i':
            testName = optarg;
            break;
        case 't':
            useThrust = true;
	    break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    int* inarray = new int[N];
    int* resultarray = new int[N];
    int* checkarray = new int[N];


    if (testName.compare("random") == 0) {

        srand(time(NULL));

        // generate random array
        for (int i = 0; i < N; i++) {
            int val = rand() % 10;
            inarray[i] = val;
            checkarray[i] = val;
        }
    } else {
        //fixed test case - you may find this useful for debugging
        srand(4); //random seed chosen by fair dice roll
        for(int i = 0; i < N; i++) {
            int val = rand() % 10;
            inarray[i] = val;
            checkarray[i] = val;
        }
    }

    printCudaInfo();

    //Test exclusive_scan
    double cudaTime = 50000.;
    // run CUDA implementation
    for (int i=0; i<3; i++) {
        if (useThrust)
            cudaTime = std::min(cudaTime, cudaScanThrust(inarray, inarray+N, resultarray));
        else
            cudaTime = std::min(cudaTime, cudaScan(inarray, inarray+N, resultarray));
    }
    printf("CUDA scan time: %.3f ms\n", 1000.f * cudaTime);

    // run reference CPU implementation
    double serialTime = 50000.;
    for (int i = 0; i < 3; i++) {
        double startTime = CycleTimer::currentSeconds();
        cpu_exclusive_scan(inarray, inarray+N, checkarray);
        double endTime = CycleTimer::currentSeconds();
        serialTime = std::min(serialTime, endTime - startTime);
    }

    printf("Serial CPU scan time: %.3f ms\n", 1000.f * serialTime);
    printf("Scan: %.3fx speedup\n", serialTime / cudaTime);

    // validate results
    for (int i = 0; i < N; i++)
    {
        if(checkarray[i] != resultarray[i])
        {
            fprintf(stderr,
                    "Error: Device exclusive_scan outputs incorrect result."
                    " A[%d] = %d, expecting %d.\n",
                    i, resultarray[i], checkarray[i]);
            exit(0);
        }
    }
    printf("Scan outputs are correct!\n\n");

    // Test find_repeats
    cudaTime = 50000.;
    // run CUDA implementation
    int cu_size;
    for (int i=0; i<3; i++) {
        cudaTime = std::min(cudaTime, 
                            cudaFindRepeats(inarray, N, resultarray, &cu_size));
    }
    printf("CUDA find_repeats time: %.3f ms\n", 1000.f * cudaTime);

    // run reference CPU implementation
    serialTime = 50000.;
    int serial_size;
    for (int i = 0; i < 3; i++) {
        double startTime = CycleTimer::currentSeconds();
        serial_size = cpu_find_repeats(inarray, N, checkarray);
        double endTime = CycleTimer::currentSeconds();
        serialTime = std::min(serialTime, endTime - startTime);
    }

    printf("Serial CPU find_repeats time: %.3f ms\n", 1000.f * serialTime);
    printf("find_repeats: %.3fx speedup\n", serialTime / cudaTime);

    // validate results
    if(serial_size != cu_size){
        fprintf(stderr, 
                "Error: Device find_repeats outputs incorrect size. "
                "Expected %d, got %d.\n",
                serial_size, cu_size);
    }
    for (int i = 0; i < serial_size; i++)
    {
        if(checkarray[i] != resultarray[i])
        {
            fprintf(stderr,
                    "Error: Device find_repeats outputs incorrect result."
                    " A[%d] = %d, expecting %d.\n",
                    i, resultarray[i], checkarray[i]);
            exit(0);
        }
    }
    printf("find_repeats outputs are correct!\n");

    delete[] inarray;
    delete[] resultarray;
    delete[] checkarray;
    return 0;
}
