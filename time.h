/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */

#pragma once

// A simple timer class

#include "cuda_runtime.h"
class timer
{
    cudaEvent_t start;
    cudaEvent_t end;

    public:
    timer()
    { 
        cudaEventCreate(&start); 
        cudaEventCreate(&end);
        cudaEventRecord(start,0);
    }

    ~timer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float milliseconds_elapsed()
    { 
        float elapsed_time;
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        return elapsed_time;
    }

    float seconds_elapsed()
    { 
        return   (milliseconds_elapsed() / 1000.0);
    }
};


