// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <gtest/gtest.h>
#include <stdio.h>

#include <random>

#include <vector>
#include <rocrand/rocrand_discrete.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define TEST_SIZE 1000000

template <typename T>
__global__ void discrete_alias_kernel(unsigned int * output, T * input, rocrand_discrete_distribution_st &dis,  const size_t N){
    auto bIdx = blockIdx.x, tIdx = threadIdx.x, bSize = blockDim.x;
    auto idx = bIdx * bSize + tIdx;

    if(idx >= N) 
        return;
    output[idx] = rocrand_device::detail::discrete_alias(input[idx], dis);
}

template <typename T>
__global__ void discrete_cdf_kernel(unsigned int * output, T * input, rocrand_discrete_distribution_st &dis,  const size_t N){
    auto bIdx = blockIdx.x, tIdx = threadIdx.x, bSize = blockDim.x;
    auto idx = bIdx * bSize + tIdx;

    if(idx >= N) 
        return;
    output[idx] = rocrand_device::detail::discrete_cdf(input[idx], dis);
}

template <typename T>
struct run_discrete_distribution_tests{
    std::vector<std::vector<double>> allProbabilities;
    std::vector<size_t> sizes;
    size_t totalSize;


    run_discrete_distribution_tests(){
        // From top to bottom:
        /**
         * discrete uniform
         * discrete normal (6 sided dice roll)
         * discrete poisson (lambda = 10)
         * discrete log normal (mean = 1, std = 5)
         */
        std::vector<std::vector<T>> aProb = {
            {10, 10, 10, 10},
            {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
            {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
            {1, 2, 8, 4, 3, 2, 1}
        };

        this->totalSize = aProb.size();

        this->allProbabilities.assign(this->totalSize, std::vector<double>());
        this->sizes.assign(this->totalSize, 0);

        for(size_t i = 0; i < this->totalSize; i++){
            this->sizes[i] = aProb[i].size();
            this->allProbabilities[i].assign(this->sizes[i], 0);
            
            T sum = 0;

            for(const T & x : aProb[i])
                sum += x;
            
            double dSum = static_cast<double>(sum);
            for(size_t ii = 0; ii < this->sizes[i]; ii++)
                this->allProbabilities[i][ii] = static_cast<double>(aProb[i][ii]) / dSum; 
        }      
    }
    
    template<typename KernelCallFunc>
    void runTestInternal(const KernelCallFunc & kcf, const size_t inputSize){
        T * hInput = new T[inputSize];
        T * dInput;

        unsigned int * hOutput = new unsigned int [inputSize];
        unsigned int * dOutput;
        
        HIP_CHECK(hipMalloc(&dInput, sizeof(T) * inputSize));
        HIP_CHECK(hipMalloc(&dOutput, sizeof(unsigned int) * inputSize));

        std::random_device                          rd;
        std::mt19937                                gen(rd());
        
        // If unsigned long is passed into the uniform distribution, it will cause
        // memory access faults
        if(std::is_same<T, unsigned long>::value){
            unsigned int maxi = std::numeric_limits<unsigned int>::max();
            unsigned int mini = std::numeric_limits<unsigned int>::min();
            std::uniform_int_distribution<unsigned int> dis(mini, maxi);

            for(size_t i = 0; i < inputSize; i++)
                hInput[i] = dis(gen);
        }
        else{
            T maxi = std::numeric_limits<T>::max();
            T mini = std::numeric_limits<T>::min();
            std::uniform_int_distribution<T> dis(mini, maxi);

            for(size_t i = 0; i < inputSize; i++)
                hInput[i] = dis(gen);
        }
        
        HIP_CHECK(hipMemcpy(dInput, hInput, sizeof(T) * inputSize, hipMemcpyHostToDevice));

        for(size_t i = 0; i < totalSize; i++){
            std::vector<double> & prob = allProbabilities[i];
            size_t N = sizes[i];

            rocrand_discrete_distribution discrete_distribution;
            HIP_CHECK(rocrand_create_discrete_distribution(prob.data(), prob.size(), 0, &discrete_distribution));

            kcf(dOutput, dInput, *discrete_distribution, inputSize);

            HIP_CHECK(hipMemcpy(hOutput, dOutput, sizeof(unsigned int) * inputSize, hipMemcpyDeviceToHost));

            std::vector<double> count(N, 0);
            for(size_t j = 0; j < inputSize; j++)
                count[hOutput[j]]++;

            for(size_t j = 0; j < N; j++){
                double res = count[j] / static_cast<double>(inputSize);
                double eps = (prob[j] >= 0.05) ? prob[j] * 0.2 : 1e-2; // within 20% if the probability is at least 5%, within 0.01 otherwise
                ASSERT_NEAR(res, prob[j], eps) << "Difference: " << std::abs(res - prob[j]) << " Epsilon: " << eps << std::endl;
            }
            
            HIP_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
        }

        HIP_CHECK(hipFree(dInput));
        HIP_CHECK(hipFree(dOutput));
    
    }

    void run_alias_test(const size_t inputSize){
        runTestInternal(
            [] __host__ __device__ (unsigned int * output, T * input, rocrand_discrete_distribution_st &dis,  const size_t N){
                size_t threads = 512;
                size_t blocks = std::ceil(static_cast<double>(N) / static_cast<double>(threads));
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(discrete_alias_kernel<T>),
                    dim3(blocks),
                    dim3(threads),
                    0,
                    0,
                    output,
                    input,
                    dis,
                    N
                );
            },
            inputSize
        );
    
    }

    void run_cdf_test(const size_t inputSize){
        runTestInternal(
            [] __host__ __device__ (unsigned int * output, T * input, rocrand_discrete_distribution_st &dis,  const size_t N){
                size_t threads = 512;
                size_t blocks = std::ceil(static_cast<double>(N) / static_cast<double>(threads));
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(discrete_cdf_kernel<T>),
                    dim3(blocks),
                    dim3(threads),
                    0,
                    0,
                    output,
                    input,
                    dis,
                    N
                );
            },
            inputSize
        );
    }
        
};

TEST(discrete_distribution_tests, discrete_alias_basic){

    std::vector<double> prob = {0.1, 0.4, 0.4, 0.1};

    rocrand_discrete_distribution discrete_distribution;
    HIP_CHECK(rocrand_create_discrete_distribution(prob.data(), prob.size(), 0, &discrete_distribution));


    size_t inputSize = 10;
    double hInput[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    unsigned int hOutput[10];
    double * dInput;
    unsigned int * dOutput;

    HIP_CHECK(hipMalloc(&dInput, sizeof(double) * inputSize));
    HIP_CHECK(hipMalloc(&dOutput, sizeof(unsigned int) * inputSize));

    HIP_CHECK(hipMemcpy(dInput, hInput, sizeof(double) * inputSize, hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(HIP_KERNEL_NAME(discrete_alias_kernel<double>),
        dim3(1),
        dim3(10),
        0,
        0,
        dOutput,
        dInput,
        *discrete_distribution,
        inputSize
    );

    HIP_CHECK(hipMemcpy(hOutput, dOutput, sizeof(unsigned int) * inputSize, hipMemcpyDeviceToHost));

    std::vector<unsigned int> rChecker(4, 0);
    for(size_t i = 0; i < inputSize; i++)
        rChecker[hOutput[i]]++;
    
    ASSERT_EQ(rChecker[0], 1);
    ASSERT_EQ(rChecker[1], 4);
    ASSERT_EQ(rChecker[2], 4);
    ASSERT_EQ(rChecker[3], 1);

    HIP_CHECK(hipFree(dInput));
    HIP_CHECK(hipFree(dOutput));
    HIP_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));

}   

TEST(discrete_distribution_tests, discrete_alias_unsigned_int){
    run_discrete_distribution_tests<unsigned int> rt;
    rt.run_alias_test(TEST_SIZE);
}

TEST(discrete_distribution_tests, discrete_alias_unsigned_long){
    run_discrete_distribution_tests<unsigned long> rt;
    rt.run_alias_test(TEST_SIZE);
}

TEST(discrete_distribution_tests, discrete_alias_unsigned_long_long){
    run_discrete_distribution_tests<unsigned long long> rt;
    rt.run_alias_test(TEST_SIZE);
}

TEST(discrete_distribution_tests, discrete_cdf_basic){

    std::vector<double> prob = {0.1, 0.4, 0.4, 0.1};

    rocrand_discrete_distribution discrete_distribution;
    HIP_CHECK(rocrand_create_discrete_distribution(prob.data(), prob.size(), 0, &discrete_distribution));

    size_t inputSize = 11;
    double hInput[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
    unsigned int hOutput[11];
    double * dInput;
    unsigned int * dOutput;

    HIP_CHECK(hipMalloc(&dInput, sizeof(double) * inputSize));
    HIP_CHECK(hipMalloc(&dOutput, sizeof(unsigned int) * inputSize));

    HIP_CHECK(hipMemcpy(dInput, hInput, sizeof(double) * inputSize, hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(HIP_KERNEL_NAME(discrete_cdf_kernel<double>),
        dim3(1),
        dim3(inputSize),
        0,
        0,
        dOutput,
        dInput,
        *discrete_distribution,
        inputSize
    );

    HIP_CHECK(hipMemcpy(hOutput, dOutput, sizeof(unsigned int) * inputSize, hipMemcpyDeviceToHost));

    std::vector<unsigned int> rChecker(4, 0);
    for(size_t i = 0; i < inputSize; i++)
        rChecker[hOutput[i]]++;
    

    ASSERT_EQ(rChecker[0], 2);
    ASSERT_EQ(rChecker[1], 4);
    ASSERT_EQ(rChecker[2], 4);
    ASSERT_EQ(rChecker[3], 1);

    HIP_CHECK(hipFree(dInput));
    HIP_CHECK(hipFree(dOutput));
    HIP_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));

}

TEST(discrete_distribution_tests, discrete_cdf_unsigned_int){
    run_discrete_distribution_tests<unsigned int> rt;
    rt.run_cdf_test(TEST_SIZE);
}

TEST(discrete_distribution_tests, discrete_cdf_unsigned_long){
    run_discrete_distribution_tests<unsigned long> rt;
    rt.run_cdf_test(TEST_SIZE);
}

TEST(discrete_distribution_tests, discrete_cdf_unsigned_long_long){
    run_discrete_distribution_tests<unsigned long long> rt;
    rt.run_cdf_test(TEST_SIZE);
}
