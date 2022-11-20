// No pragma once, we want it multiple times!

#include <benchmark/benchmark.h>
#include <funzel/Tensor.hpp>

#ifndef CommonTest
#define CommonTest ""
#endif

#ifndef TestDevice
#define TestDevice ""
#endif

#ifndef TestDtype
#define TestDtype DFLOAT32
#endif

#define PREFIXED_BENCHMARK(b) \
	BENCHMARK(b)->Name(CommonTest #b)

using namespace funzel;

static void Empty(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t;
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t = Tensor::empty({sz}, TestDtype, TestDevice));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(state.range(0));
	state.counters["ValuesRate"] = benchmark::Counter(state.range(0), benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(Empty)->Range(1 << 10, 1 << 30)->Complexity();

static void Ones(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t;
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t = Tensor::ones({sz}, TestDtype, TestDevice));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(state.range(0));
	state.counters["ValuesRate"] = benchmark::Counter(state.range(0), benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(Ones)->Range(1 << 10, 1 << 30)->Complexity();

static void Zeros(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t;
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t = Tensor::zeros({sz}, TestDtype, TestDevice));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(state.range(0));
	state.counters["ValuesRate"] = benchmark::Counter(state.range(0), benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(Zeros)->Range(1 << 10, 1 << 30)->Complexity();

static void Matmul(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t1 = Tensor::ones({sz, sz}, TestDtype, TestDevice);
	Tensor t2 = Tensor::ones({sz, sz}, TestDtype, TestDevice);
	
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t1.matmul(t2));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(2*(sz*sz));
	state.counters["ValuesRate"] = benchmark::Counter(2*(sz*sz), benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(Matmul)->Range(1 << 5, 1 << 13)->Complexity();

static void MatmulMatrixVector(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t1 = Tensor::ones({sz, sz}, TestDtype, TestDevice);
	Tensor t2 = Tensor::ones({sz, 1}, TestDtype, TestDevice);
	
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t1.matmul(t2));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(sz*sz + sz);
	state.counters["ValuesRate"] = benchmark::Counter(sz*sz + sz, benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(MatmulMatrixVector)->Range(1 << 5, 1 << 15)->Complexity();

static void MatmulVectorVector(benchmark::State& state)
{
	const size_t sz = state.range(0);

	Tensor t1 = Tensor::ones({1, sz}, TestDtype, TestDevice);
	Tensor t2 = Tensor::ones({sz, 1}, TestDtype, TestDevice);
	
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t1.matmul(t2));
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(2*sz);
	state.counters["ValuesRate"] = benchmark::Counter(2*sz, benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(MatmulVectorVector)->Range(1 << 5, 1 << 15)->Complexity();

static void Abs(benchmark::State& state)
{
	const size_t sz = state.range(0);
	Tensor t = Tensor::ones({sz}, TestDtype, TestDevice);
	
	for (auto _ : state)
	{
		benchmark::DoNotOptimize(t.abs_());
		benchmark::ClobberMemory();
	}
	state.SetComplexityN(sz);
	state.counters["ValuesRate"] = benchmark::Counter(sz, benchmark::Counter::kIsRate);
}
PREFIXED_BENCHMARK(Abs)->Range(1 << 5, 1 << 25);//->Complexity();
