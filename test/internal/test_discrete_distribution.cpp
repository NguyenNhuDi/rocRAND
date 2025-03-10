// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rng/distribution/normal.hpp>
#include <rocrand/rocrand_mtgp32_11213.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
using namespace rocrand_impl::host;

TEST(normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t               size = 4000;
    float                      val[size];
    normal_distribution<float> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                size = 4000;
    double                      val[size];
    normal_distribution<double> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[4];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        input[2] = dis(gen);
        input[3] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(normal_distribution_tests, float_out_uint2_in_test)
{

    struct nd
    {
        const float mean;
        const float stddev;

        nd(float mean, float stddev) : mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        void operator()(const uint2 &input, float (&output)[2]) const
        {
            float2 v  = rocrand_device::detail::normal_distribution2(input);
            output[0] = mean + v.x * stddev;
            output[1] = mean + v.y * stddev;
        }
    };

    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        val[size];
    nd           u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        uint2     input;
        float     output[2];
        input.x = dis(gen);
        input.y = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += (output[0] + output[1]);
    }
    mean /= size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}


TEST(normal_distribution_tests, float2_out_longlong_in_test)
{
    
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        val[size];
    normal_distribution<float, unsigned long long> u(2, 5);
    
    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned long long input[1];
        float    output[2];
        unsigned long long l = static_cast<unsigned long long>(dis(gen));
        unsigned long long r = static_cast<unsigned long long>(dis(gen));
        input[0] = (l << 32) | r;
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += (output[0] + output[1]);
    }
    mean /= size;
    
    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2) / size;
    }
    std = std::sqrt(std);
    
    EXPECT_NEAR(2, mean, 0.4) << "Mean: " << mean << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, std, 1.0) <<  "Stddev: " << std << " Expected: " << 5; // 20%
}

TEST(normal_distribution_tests, float4_outputs){
    struct normal_distribution_float_4_out{
        const float mean;
        const float stddev;

        normal_distribution_float_4_out(float mean, float stddev) : mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        void uint4_in(const uint4 &input, float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }

        __forceinline__ __host__ __device__
        void longlong2_in(const longlong2 &input, float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }

        __forceinline__ __host__ __device__
        void ull_2_in(const unsigned long long (&input)[2], float (&output)[4]) const
        {
            float4 v  = rocrand_device::detail::normal_distribution4(input[0], input[1]);
            output[0] = mean + v.w * stddev; 
            output[1] = mean + v.x * stddev;
            output[2] = mean + v.y * stddev;
            output[3] = mean + v.z * stddev;
        }
    };
    
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float        vull2[size], vll2[size], vui4[size];
    normal_distribution_float_4_out u(2, 5);

    // Calculate mean
    double mui4 = 0, mll2 = 0, mull2 = 0;
    for(size_t i = 0; i < size; i += 4)
    {
        unsigned long long ull2[2];
        longlong2          ll2;
        uint4              ui4;
        float     output[4];
        unsigned long long l = static_cast<unsigned long long>(dis(gen));
        unsigned long long r = static_cast<unsigned long long>(dis(gen));
        
        unsigned long long in1 = (l << 32) | r;
    
        ull2[0] = in1;
        ll2.x = in1;
        ui4.w = l; ui4.x = r;

        l = static_cast<unsigned long long>(dis(gen));
        r = static_cast<unsigned long long>(dis(gen));
        unsigned long long in2 = (l << 32) | r;

        ull2[1] = in2;
        ll2.y = in2;
        ui4.y = l; ui4.z = r;

        u.uint4_in(ui4, output);
        vui4[i] = output[0]; 
        vui4[i + 1] = output[1]; 
        vui4[i + 2] = output[2]; 
        vui4[i + 3] = output[3];
        mui4 += (output[0] + output[1] + output[2] + output[3]) / size;

        u.longlong2_in(ll2, output);
        vll2[i] = output[0]; 
        vll2[i + 1] = output[1]; 
        vll2[i + 2] = output[2]; 
        vll2[i + 3] = output[3];
        mll2 += (output[0] + output[1] + output[2] + output[3]) / size;

        u.ull_2_in(ull2, output);
        vull2[i] = output[0]; 
        vull2[i + 1] = output[1]; 
        vull2[i + 2] = output[2]; 
        vull2[i + 3] = output[3];
        mull2 += (output[0] + output[1] + output[2] + output[3]) / size;
    }

    double sui4 = 0, sll2 = 0, sull2 = 0;
    for(size_t i = 0; i < size; i++)
    {
        sui4 += std::pow(vui4[i] - mui4, 2);
        sll2 += std::pow(vll2[i] - mll2, 2);
        sull2 += std::pow(vull2[i] - mull2, 2);
    }

    sui4 = sqrt(sui4 / size);
    sll2 = sqrt(sll2 / size);
    sull2 = sqrt(sull2 / size);

    EXPECT_NEAR(2, mui4, 0.4) << "Mean: " << mui4 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sui4, 1.0) <<  "Stddev: " << sui4 << " Expected: " << 5; // 20%

    EXPECT_NEAR(2, mll2, 0.4) << "Mean: " << mll2 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sll2, 1.0) <<  "Stddev: " << sll2 << " Expected: " << 5; // 20%

    EXPECT_NEAR(2, mull2, 0.4) << "Mean: " << mull2 << " Expected: " << 2; // 20%
    EXPECT_NEAR(5, sull2, 1.0) <<  "Stddev: " << sull2 << " Expected: " << 5; // 20%
}

TEST(normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t              size = 4000;
    half                      val[size];
    normal_distribution<half> u(__float2half(2.0f), __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

template<typename mrg, unsigned int m1>
struct mrg_normal_distribution_test_type
{
    typedef mrg                   mrg_type;
    static constexpr unsigned int mrg_m1 = m1;
};

template<typename test_type>
struct mrg_normal_distribution_tests : public ::testing::Test
{
    typedef typename test_type::mrg_type mrg_type;
    static constexpr unsigned int        mrg_m1 = test_type::mrg_m1;
};

typedef ::testing::Types<
    mrg_normal_distribution_test_type<rocrand_state_mrg31k3p, ROCRAND_MRG31K3P_M1>,
    mrg_normal_distribution_test_type<rocrand_state_mrg32k3a, ROCRAND_MRG32K3A_M1>>
    mrg_normal_distribution_test_types;

TYPED_TEST_SUITE(mrg_normal_distribution_tests, mrg_normal_distribution_test_types);

TYPED_TEST(mrg_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                          size = 4000;
    float                                                                 val[size];
    mrg_engine_normal_distribution<float, typename TestFixture::mrg_type> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TYPED_TEST(mrg_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                           size = 4000;
    double                                                                 val[size];
    mrg_engine_normal_distribution<double, typename TestFixture::mrg_type> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TYPED_TEST(mrg_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                         size = 4000;
    half                                                                 val[size];
    mrg_engine_normal_distribution<half, typename TestFixture::mrg_type> u(__float2half(2.0f),
                                                                           __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(sobol_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                     size = 4000;
    float                            val[size];
    sobol_normal_distribution<float> u(2.0f, 5.0f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        float        output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(sobol_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                      size = 4000;
    double                            val[size];
    sobol_normal_distribution<double> u(2.0, 5.0);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        double       output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%
}

TEST(sobol_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                    size = 4000;
    half                            val[size];
    sobol_normal_distribution<half> u(__float2half(2.0f), __float2half(5.0f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        unsigned int input[1];
        half         output[1];
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        val[i]    = output[0];
        mean += __half2float(output[0]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

template <typename State>
void run_normal_dist_with_state_out1(State * states){
    double expected_mean = 2.0f, expected_std = 5.0f;

    struct single_out{
        double mean, std;
        State * states;
        
        single_out(double mean, double std, State * states){
            this->mean = mean;
            this->std = std;
            this->states = states;
        }

        float __forceinline__ __device__ __host__ fOp(){
            return mean + rocrand_normal(states) * std;
        }

        double __forceinline__ __device__ __host__ dOp(){
            return mean + rocrand_normal_double(states) * std;
        }
    };

    const size_t size = 4000;

    float fOut[size];
    double dOut[size];

    single_out s(expected_mean, expected_std, states);

    float fMean = 0, fStd = 0;
    double dMean = 0, dStd = 0;

    for(size_t i = 0; i < size; i++){
        fOut[i] = s.fOp();
        dOut[i] = s.dOp();

        fMean += fOut[i] / size;
        dMean += dOut[i] / size;
    }
    
    for(size_t i = 0; i < size; i++){
        fStd += std::pow(fOut[i] - fMean, 2) / size;
        dStd += std::pow(dOut[i] - dMean, 2) / size;
    }

    fStd = std::sqrt(fStd);
    dStd = std::sqrt(dStd);

    EXPECT_NEAR(expected_mean, fMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << fMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, fStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << fStd << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, dMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << dMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, dStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << dStd << " Expected: " << expected_std; // 20%
}

template <typename State>
void run_normal_dist_with_state_out2(State * states){
    double expected_mean = 2.0f, expected_std = 5.0f;
    struct nd{
        double mean, std;
        State * states;
        nd(double  mean, double stddev, State * states) {
            this->mean = mean;
            this->std = stddev;
            this->states = states;
        }
        __forceinline__ __host__ __device__ void operator()(float(&output)[2]){
            float2 v = rocrand_normal2(states);
            output[0] = static_cast<float>(mean) + v.x * static_cast<float>(std);
            output[1] = static_cast<float>(mean) + v.y * static_cast<float>(std);
        }
        
        __forceinline__ __host__ __device__ void operator()(double(&output)[2]){
            double2 v = rocrand_normal_double2(states);
            output[0] = mean + v.x * std;
            output[1] = mean + v.y * std;
        }
    };
    const size_t size = 4000;
    float valF[size];
    double valD[size];
    nd u(expected_mean, expected_std, states);

    double meanF = 0, meanD = 0;
    for(size_t i = 0; i < size; i += 2)
    {
        float fOut[2];
        double dOut[2];
        
        u(fOut);
        valF[i]     = fOut[0];
        valF[i + 1] = fOut[1];
        meanF += (fOut[0] + fOut[1]) / size;

        u(dOut);
        valD[i]     = dOut[0];
        valD[i + 1] = dOut[1];
        meanD += (dOut[0] + dOut[1]) / size;
    }

    // Calculate stddev
    double stdF = 0, stdD = 0;
    for(size_t i = 0; i < size; i++){
        stdF += std::pow(valF[i] - meanF, 2) / size;
        stdD += std::pow(valD[i] - meanD, 2) / size;
    }
    stdF = std::sqrt(stdF);
    stdD = std::sqrt(stdD);

    EXPECT_NEAR(expected_mean, meanF, (expected_mean * 0.2) + 1e-1) << "Mean: " << meanF << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, stdF, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << stdF << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, meanD, (expected_mean * 0.2) + 1e-1) << "Mean: " << meanD << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, stdD, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << stdD << " Expected: " << expected_std; // 20%
}

template <typename State>
void run_normal_dist_with_state_out4(State * states){
    double expected_mean = 2.0f, expected_std = 5.0f;
    
    struct nd{
        double mean, std;
        State * states;
        nd(double  mean, double stddev, State * states) {
            this->mean = mean;
            this->std = stddev;
            this->states = states;
        }
        __forceinline__ __host__ __device__ void operator()(float(&output)[4]){
            float4 v = rocrand_normal4(states);
            output[0] = static_cast<float>(mean) + v.w * static_cast<float>(std);
            output[1] = static_cast<float>(mean) + v.x * static_cast<float>(std);
            output[2] = static_cast<float>(mean) + v.y * static_cast<float>(std);
            output[3] = static_cast<float>(mean) + v.z * static_cast<float>(std);
        }
        
        __forceinline__ __host__ __device__ void operator()(double(&output)[4]){
            double4 v = rocrand_normal_double4(states);
            output[0] = mean + v.w * std;
            output[1] = mean + v.x * std;
            output[2] = mean + v.y * std;
            output[3] = mean + v.z * std;
        }
    };
    const size_t size = 4000;
    float valF[size];
    double valD[size];

    nd u(expected_mean, expected_std, states);

    double meanF = 0, meanD = 0;
    for(size_t i = 0; i < size; i += 4)
    {
        float fOut[4];
        double dOut[4];
        
        u(fOut);
        valF[i]     = fOut[0];
        valF[i + 1] = fOut[1];
        valF[i + 2] = fOut[2];
        valF[i + 3] = fOut[3];
        meanF += (fOut[0] + fOut[1] + fOut[2] + fOut[3]) / size;

        u(dOut);
        valD[i]     = dOut[0];
        valD[i + 1] = dOut[1];
        valD[i + 2] = dOut[2];
        valD[i + 3] = dOut[3];
        meanD += (dOut[0] + dOut[1] + dOut[2] + dOut[3]) / size;
    }

    // Calculate stddev
    double stdF = 0, stdD = 0;
    for(size_t i = 0; i < size; i++){
        stdF += std::pow(valF[i] - meanF, 2) / size;
        stdD += std::pow(valD[i] - meanD, 2) / size;
    }
    stdF = std::sqrt(stdF);
    stdD = std::sqrt(stdD);

    EXPECT_NEAR(expected_mean, meanF, (expected_mean * 0.2) + 1e-2) << "Mean: " << meanF << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, stdF, (expected_std * 0.2) + 1e-2) <<  "Stddev: " << stdF << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, meanD, (expected_mean * 0.2) + 1e-2) << "Mean: " << meanD << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, stdD, (expected_std * 0.2) + 1e-2) <<  "Stddev: " << stdD << " Expected: " << expected_std; // 20%
}
    
TEST(normal_distribution_with_states, philox4x32_10){
    rocrand_state_philox4x32_10 states;
    rocrand_init(123456, 654321, 0, &states);
    run_normal_dist_with_state_out2<rocrand_state_philox4x32_10>(&states);
    run_normal_dist_with_state_out4<rocrand_state_philox4x32_10>(&states);
}

TEST(normal_distribution_with_states, mrg31k3p){
    rocrand_state_mrg31k3p states;
    rocrand_init(123456, 654321, 0, &states);
    run_normal_dist_with_state_out2<rocrand_state_mrg31k3p>(&states);
}

TEST(normal_distribution_with_states, mrg32k3a){
    rocrand_state_mrg32k3a states;
    rocrand_init(123456, 654321, 0, &states);
    run_normal_dist_with_state_out2<rocrand_state_mrg32k3a>(&states);
}

TEST(normal_distribution_with_states, xorwow){
    rocrand_state_xorwow states;
    rocrand_init(123456, 654321, 0, &states);
    run_normal_dist_with_state_out2<rocrand_state_xorwow>(&states);
}

TEST(normal_distribution_with_states, sobol32){
    rocrand_state_sobol32 states;
    const unsigned int* directions;
    HIP_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

    rocrand_init(directions, 0, &states);

    run_normal_dist_with_state_out1<rocrand_state_sobol32>(&states);

}

TEST(normal_distribution_with_states, scarambled_sobol32){
    rocrand_state_scrambled_sobol32 states;
    const unsigned int* directions;
    HIP_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

    rocrand_init(directions, 123456, 0, &states);

    run_normal_dist_with_state_out1<rocrand_state_scrambled_sobol32>(&states);
}

TEST(normal_distribution_with_states, sobol64){
    rocrand_state_sobol64 states;
    const unsigned long long* directions;
    HIP_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));

    rocrand_init(directions, 0, &states);

    run_normal_dist_with_state_out1<rocrand_state_sobol64>(&states);

}

TEST(normal_distribution_with_states, scarambled_sobol64){
    rocrand_state_scrambled_sobol64 states;
    const unsigned long long* directions;
    HIP_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));

    rocrand_init(directions, 123456, 0, &states);

    run_normal_dist_with_state_out1<rocrand_state_scrambled_sobol64>(&states);
}


TEST(normal_distribution_with_states, lfsr113){
    rocrand_state_lfsr113 states;
    rocrand_init(static_cast<uint4>(12), 0, &states);

    run_normal_dist_with_state_out1<rocrand_state_lfsr113>(&states);
    run_normal_dist_with_state_out2<rocrand_state_lfsr113>(&states);
}

TEST(normal_distribution_with_states, threefry2x32_20){
    rocrand_state_threefry2x32_20 states;
    rocrand_init(123456, 654321, 0, & states);

    run_normal_dist_with_state_out1<rocrand_state_threefry2x32_20>(&states);
    run_normal_dist_with_state_out2<rocrand_state_threefry2x32_20>(&states);
}

TEST(normal_distribution_with_states, threefry2x64_20){
    rocrand_state_threefry2x64_20 states;
    rocrand_init(123456, 654321, 0, & states);

    run_normal_dist_with_state_out1<rocrand_state_threefry2x64_20>(&states);
    run_normal_dist_with_state_out2<rocrand_state_threefry2x64_20>(&states);
}

TEST(normal_distribution_with_states, rocrand_state_threefry4x32_20){
    rocrand_state_threefry4x32_20 states;
    rocrand_init(123456, 654321, 0, & states);

    run_normal_dist_with_state_out1<rocrand_state_threefry4x32_20>(&states);
    run_normal_dist_with_state_out2<rocrand_state_threefry4x32_20>(&states);
}

TEST(normal_distribution_with_states, rocrand_state_threefry4x64_20){
    rocrand_state_threefry4x64_20 states;
    rocrand_init(123456, 654321, 0, & states);

    run_normal_dist_with_state_out1<rocrand_state_threefry4x64_20>(&states);
    run_normal_dist_with_state_out2<rocrand_state_threefry4x64_20>(&states);
}
__global__ void mtgp32_run_rocrand_normal(rocrand_state_mtgp32 * state, float * fout){
    fout[0] = rocrand_normal(state);
}

__global__ void mtgp32_run_rocrand_normal2(rocrand_state_mtgp32 * state, float * fout){
    float2 f2 = rocrand_normal2(state);
    fout[1] = f2.x;
    fout[2] = f2.y;
}

__global__ void mtgp32_run_rocrand_normal_double(rocrand_state_mtgp32 * state, double * dout){
    dout[0] = rocrand_normal_double(state);
}

__global__ void mtgp32_run_rocrand_normal_double2(rocrand_state_mtgp32 * state, double * dout){
    double2 d2 = rocrand_normal_double2(state);
    dout[1] = d2.x;
    dout[2] = d2.y;
}

TEST(normal_distribution_with_states, mtgp32){
    const size_t size = 6000;
    double expected_mean = 2, expected_std = 5;

    rocrand_state_mtgp32 * states;
    auto state_size = 123, seed = 123456;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    float * fOut; double * dOut;

    HIP_CHECK(hipMalloc(&fOut, 3 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut, 3 * sizeof(double)));

    float singleFloats[size];
    float doubleFloats[size * 2 ];
    double singleDoubles[size];
    double doubleDoubles[size * 2];

    float sFloatMean = 0, dFloatMean = 0;
    double sDoubleMean = 0, dDoubleMean = 0;

    size_t dIndex = 0;
    for(size_t i = 0; i < size; i ++){
        mtgp32_run_rocrand_normal<<<1, 1>>>(states, fOut);
        mtgp32_run_rocrand_normal2<<<1, 1>>>(states, fOut);

        mtgp32_run_rocrand_normal_double<<<1, 1>>>(states, dOut);
        mtgp32_run_rocrand_normal_double2<<<1, 1>>>(states, dOut);
        
        float fTemp[3]; double dTemp[3];
        
        HIP_CHECK(hipMemcpy(fTemp, fOut, 3 * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(dTemp, dOut, 3 * sizeof(double), hipMemcpyDeviceToHost));

        singleFloats[i] = expected_mean + fTemp[0] * expected_std;
        doubleFloats[dIndex] = expected_mean + fTemp[1] * expected_std;
        doubleFloats[dIndex + 1] = expected_mean + fTemp[2] * expected_std;

        singleDoubles[i] = expected_mean + dTemp[0] * expected_std;
        doubleDoubles[dIndex] = expected_mean + dTemp[1] * expected_std;
        doubleDoubles[dIndex + 1] = expected_mean + dTemp[2] * expected_std;

        sFloatMean += singleFloats[i]; dFloatMean += doubleFloats[dIndex] + doubleFloats[dIndex + 1];
        sDoubleMean += singleDoubles[i]; dDoubleMean += doubleDoubles[dIndex] + doubleDoubles[dIndex + 1];

        dIndex += 2;
    }

    sFloatMean /= size; dFloatMean /= 2 * size;
    sDoubleMean /= size; dDoubleMean /= 2 * size;

    // Calculate stddev
    
    float sFloatStd = 0, dFloatStd = 0;
    double sDoubleStd = 0, dDoubleStd = 0;
    dIndex = 0;
    for(size_t i = 0; i < size; i++){
        sFloatStd += std::pow(singleFloats[i] - sFloatMean, 2) / size;
        dFloatStd += std::pow(doubleFloats[dIndex] - dFloatMean, 2) / (size * 2);
        dFloatStd += std::pow(doubleFloats[dIndex + 1] - dFloatMean, 2) / (size * 2);
        
        sDoubleStd += std::pow(singleDoubles[i] - sDoubleMean, 2) / size;
        dDoubleStd += std::pow(doubleDoubles[dIndex] - dDoubleMean, 2) / (size * 2);
        dDoubleStd += std::pow(doubleDoubles[dIndex + 1] - dDoubleMean, 2) / (size * 2);

        dIndex += 2;
    }
    
    sFloatStd = std::sqrt(sFloatStd);
    dFloatStd = std::sqrt(dFloatStd);
    sDoubleStd = std::sqrt(sDoubleStd);
    dDoubleStd = std::sqrt(dDoubleStd);

    EXPECT_NEAR(expected_mean, sFloatMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << sFloatMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, sFloatStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << sFloatStd << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, dFloatMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << dFloatMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, dFloatStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << dFloatStd << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, sDoubleMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << sDoubleMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, sDoubleStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << sDoubleStd << " Expected: " << expected_std; // 20%

    EXPECT_NEAR(expected_mean, dDoubleMean, (expected_mean * 0.2) + 1e-1) << "Mean: " << dDoubleMean << " Expected: " << expected_mean; // 20%
    EXPECT_NEAR(expected_std, dDoubleStd, (expected_std * 0.2) + 1e-1) <<  "Stddev: " << dDoubleStd << " Expected: " << expected_std; // 20%

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(fOut));
    HIP_CHECK(hipFree(dOut));
}

