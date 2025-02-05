#include "benchmark_rocrand_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include "custom_csv_formater.hpp"
#include <fstream>
#include <hip/hip_runtime.h>
#include <map>
#include <rocrand/rocrand.h>
#include <string>
#include <vector>
#include <climits>
#include <random>
#include <cxxabi.h>
#include <typeinfo>

#define TEST_SIZE 1000000000

template <typename T>
T * populate(size_t size){
    T * out = new T[size];
    for(size_t i = 0; i < size; i++) out[i] = static_cast<T>(i);
    return  out;
        
}

template <typename T>
void run_benchmark(benchmark::State& state, void (*func)(T *& arr, size_t), size_t size, bool pDevice) {

    T * hData = populate<T>(size);
    T * dData;

    if(pDevice){
        hipMalloc(&dData, size * sizeof(T));
        hipMemcpy(hData, dData, sizeof(T) * size, hipMemcpyHostToDevice);
    }

    // Warm-up
    
    if(pDevice) 
        func(dData, size);
    else
        func(hData, size);


    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t start, stop;

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for(auto _ : state)
        {
            HIP_CHECK(hipEventRecord(start, stream));

            if(pDevice) 
                func(dData, size);
            else
                func(hData, size);
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));

            state.SetIterationTime(elapsed / 1000.f);
        }

    state.SetBytesProcessed((long long)state.iterations() * size * sizeof(T));
    state.SetItemsProcessed((long long)state.iterations() * size);

    delete [] hData;
    if(pDevice) hipFree(dData);
}

template <typename T>
std::string getTypeName() {
    int status;
    char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);

    return std::string(demangled);
}

template <typename T>
void populate_benchmark(std::vector<benchmark::internal::Benchmark*> &benchmarks){

    benchmarks.emplace_back(benchmark::RegisterBenchmark(
        ("hipMemcpyHostToDevice<" + getTypeName<T>() + ">--" + std::to_string(TEST_SIZE)).c_str(),
        [=](benchmark::State& st) { 
            run_benchmark<T>(
                st, 
                [](T *& hData, size_t size)
                {
                    T * dData; 
                    hipMalloc(&dData, size * sizeof(T));
                    hipMemcpy(hData, dData, sizeof(T) * size, hipMemcpyHostToDevice);
                    hipFree(dData);
                },
                TEST_SIZE,
                false
            ); 
        }
    ));


    benchmarks.emplace_back(benchmark::RegisterBenchmark(
        ("hipMemcpyDeviceToHost<" + getTypeName<T>() + ">--" + std::to_string(TEST_SIZE)).c_str(),
        [=](benchmark::State& st) { 
            run_benchmark<T>(
                st, 
                [](T *& dData, size_t size)
                {
                    T * hData = new T[size];
                    hipMemcpy(hData, dData, sizeof(T) * size, hipMemcpyDeviceToHost);
                    delete [] hData;
                },
                TEST_SIZE,
                true
            ); 
        }
    ));

}

int main(int argc, char ** argv){

    // get paramaters before they are passed into
    std::string outFormat     = "";
    std::string filter        = "";
    std::string consoleFormat = "";

    getFormats(argc, argv, outFormat, filter, consoleFormat);

    benchmark::Initialize(&argc, argv);


    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    // for(size_t i = 1; i < 100000; i *= 10){

    populate_benchmark<unsigned int>(benchmarks);
    populate_benchmark<int>(benchmarks);
    populate_benchmark<unsigned char>(benchmarks);
    populate_benchmark<char>(benchmarks);
    populate_benchmark<unsigned short>(benchmarks);
    populate_benchmark<short>(benchmarks);
    populate_benchmark<__half>(benchmarks);
    populate_benchmark<float>(benchmarks);
    populate_benchmark<double>(benchmarks);

    

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    benchmark::BenchmarkReporter* console_reporter  = getConsoleReporter(consoleFormat);
    benchmark::BenchmarkReporter* out_file_reporter = getOutFileReporter(outFormat);

    std::string spec = (filter == "" || filter == "all") ? "." : filter;

    // Run benchmarks
    if(outFormat == "") // default case
        benchmark::RunSpecifiedBenchmarks(console_reporter, spec);
    else
        benchmark::RunSpecifiedBenchmarks(console_reporter, out_file_reporter, spec);
}