extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"

#define TB 128

/* operations */
struct opPlus {
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x + y;
	}
};

struct opMinus {
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x - y;
	}
};

struct opMult {
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x * y;
	}
};

struct opDiv {
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x / y;
	}
};

struct opMax {
	static const float base_value = -2e38;
	__device__ float operator()(float x, float y)
	{
		return fmaxf(x, y);
	}
};

struct opExp {
public:
	__device__ float operator()(float x)
	{
		return exp(x);
	}
};

struct opSigmoid {
public:
	__device__ float operator()(float x)
	{
		return 1 / (1 + exp(-x));
	}
};

struct opSigmoidDeriv {
public:
	__device__ float operator()(float x, float y)
	{
		return x * y * (1 - y);
	}
};

struct opTanh {
public:
	__device__ float operator()(float x)
	{
		return tanh(x);
	}
};

struct opTanhDeriv {
public:
	__device__ float operator()(float x, float y)
	{
		return x * (1 - y * y);
	}
};

struct opCCE {
public:
	__device__ float operator()(float input, float target)
	{
		return target > 0 ? target * log(input) : 0;
	}
};

struct opHuber {
	float threshold;
	opHuber(float threshold_) : threshold(threshold_) {};
	__device__ float operator()(float x, float y) {
		float d = x - y;
		if (-threshold < d && d < threshold) {
			return 0.5 * d * d;
		} else {
			return threshold * (abs(d) - 0.5 * threshold);
		}
	}
};

struct opHuberDeriv {
	float threshold;
	opHuberDeriv(float threshold_) : threshold(threshold_) {};
	__device__ float operator()(float x, float y) {
		float d = x - y;
		if (-threshold < d && d < threshold) {
			return d;
		} else {
			return threshold * signbit(d);
		}
	}
};

struct opShrink {
	float threshold;
	opShrink(float threshold_) : threshold(threshold_) {};
	
	__device__ float operator()(float x) { 
		if (x - threshold > 0) {
			return x - threshold;
		} else if (x + threshold < 0) {
			return x + threshold;
		} else {
			return 0.0;
		}
	}
};

/* Is A in row major format? */
int is_rm(THCudaTensor *A)
{
	return A->stride[1] == 1 || A->nDimension == 1;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

/* res[i] = A[inds[i]] */
__global__ void get_cols(float *A, int A_stride, float *inds, float *res, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		res[i] = A[i * A_stride + (int)inds[i] - 1];
	}
}

int get_cols(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *res = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int len = THCudaTensor_nElement(inds);
	get_cols<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[0], THCudaTensor_data(inds), THCudaTensor_data(res), len);
	return 0;
}


/* A[inds[i]] = val */
__global__ void set_cols(float *A, int A_stride, float *inds, float val, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		A[i * A_stride + (int)inds[i] - 1] = val;
	}
}

int set_cols(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	float val = luaL_checknumber(L, 3);
	int len = THCudaTensor_nElement(inds);
	set_cols<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[0], THCudaTensor_data(inds), val, len);
	return 0;
}


template<class Op>
int transform1(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);

	if (!is_rm(A) || !is_rm(B)) {
		luaL_error(L, "Matrices not in row major order");
	}

	if (lenA != lenB) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::transform(pA, pA + lenA, pB, op);
	return 0;
}

template<class Op>
int transform2(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);
	int lenC = THCudaTensor_nElement(C);

	if (!is_rm(A) || !is_rm(B) || !is_rm(C)) {
		luaL_error(L, "Matrices not in roj major order");
	}

	if (lenA != lenB || lenA != lenC) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::device_ptr<float> pC(THCudaTensor_data(C));
	thrust::transform(pA, pA + lenA, pB, pC, op);
	return 0;
}

int huber(lua_State *L)
{
	float threshold = luaL_checknumber(L, 4);
	return transform2(opHuber(threshold), L);
}

int huber_deriv(lua_State *L)
{
	float threshold = luaL_checknumber(L, 4);
	return transform2(opHuberDeriv(threshold), L);
}

int shrink(lua_State *L)
{
	float threshold = luaL_checknumber(L, 3);
	return transform1(opShrink(threshold), L);
}

int sigmoid(lua_State *L)
{
	return transform1(opSigmoid(), L);
}

int mult_by_sigmoid_deriv(lua_State *L)
{
	return transform2(opSigmoidDeriv(), L);
}

int tanh(lua_State *L)
{
	return transform1(opTanh(), L);
}

int mult_by_tanh_deriv(lua_State *L)
{
	return transform2(opTanhDeriv(), L);
}

int cce(lua_State *L)
{
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	transform2(opCCE(), L);
	thrust::device_ptr<float> pC(THCudaTensor_data(C));
	float sum = thrust::reduce(pC, pC + THCudaTensor_nElement(C));

	lua_pushnumber(L, -sum);
	return 1;
}

int _exp(lua_State *L)
{
	return transform1(opExp(), L);
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op, int axis>
__global__ void kMatVect(Op op, float *A, float *x, float *B, int len, int size0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		if (axis == 0) B[i] = op(A[i], x[i % size0]);
		if (axis == 1) B[i] = op(A[i], x[i / size0]);
	}
}

template <class Op>
int mat_vect(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int axis = luaL_checkint(L, 4) - 1;

	assert(axis == 0 || axis == 1);

	if (!is_rm(A) || !is_rm(B)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (THCudaTensor_nElement(A) != THCudaTensor_nElement(B)) {
		luaL_error(L, "Size mismatch");
	}

	int len = THCudaTensor_nElement(A);
	if (axis == 0) {
		if (A->size[1] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 0><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), THCudaTensor_data(B), len, A->size[1]);
	} else if (axis == 1) {
		if (A->size[0] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 1><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), THCudaTensor_data(B), len, A->size[1]);
	}

	checkCudaError(L);
	return 0;
}

int add_mat_vect(lua_State *L)
{
	return mat_vect(opPlus(), L);
}

int sub_mat_vect(lua_State *L)
{
	return mat_vect(opMinus(), L);
}

int mult_mat_vect(lua_State *L)
{
	return mat_vect(opMult(), L);
}

int div_mat_vect(lua_State *L)
{
	return mat_vect(opDiv(), L);
}

__global__ void kAdd(float *A, float *B, float *C, float alpha, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) C[i] = A[i] + alpha * B[i];
}

/* C = A + alpha * B */
int add(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	float alpha = luaL_optnumber(L, 4, 1.0);

	if (!is_rm(A) || !is_rm(B) || !is_rm(C)) {
		luaL_error(L, "Matrices not in row major order");
	}

	if (!(A->size[0] == B->size[0] && A->size[1] == B->size[1] && A->size[0] == C->size[0] && A->size[1] == C->size[1])) {
		luaL_error(L, "Size mismatch");
	}

	int len = THCudaTensor_nElement(A);
	kAdd<<<(len - 1) / TB + 1, TB>>>(THCudaTensor_data(A), THCudaTensor_data(B), THCudaTensor_data(C), alpha, len);
	checkCudaError(L);
	return 0;
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op>
__global__ void kReduce(Op op, float *A, float *x, int n, int axis)
{
	extern __shared__ float sdata[];

	int i = threadIdx.x;

	sdata[i] = op.base_value;
	if (i < n) {
		if (axis == 0) {
			sdata[i] = A[gridDim.x * threadIdx.x + blockIdx.x];
		} else if (axis == 1) {
			sdata[i] = A[threadIdx.x + n * blockIdx.x];
		}
	}
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (i < s) {
			sdata[i] = op(sdata[i], sdata[i + s]);
		}
		__syncthreads();
	}

	if (i == 0) {
		x[blockIdx.x] = sdata[0];
	}
}

template <class Op>
int reduce(Op op, lua_State *L)
{
	int reduce_dim, other_dim;

	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int axis = luaL_checkint(L, 3) - 1;

	if (!is_rm(A)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (axis != 0 && axis != 1) {
		luaL_error(L, "axis not in {0, 1}");
	}

	if (axis == 0) {
		reduce_dim = A->size[0];
		other_dim = A->size[1];
	} else if (axis == 1) {
		reduce_dim = A->size[1];
		other_dim = A->size[0];
	}

	assert(reduce_dim <= 1024);
	if (other_dim != THCudaTensor_nElement(x)) {
		luaL_error(L, "Size mismatch"); 
	}

	int threads = 1;
	while(threads < reduce_dim) {
		threads = threads << 1;
	}

	kReduce<Op><<<other_dim, threads, threads * sizeof(float)>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), reduce_dim, axis);
	checkCudaError(L);
	return 0;
}

int sum(lua_State *L)
{
	return reduce(opPlus(), L);
}

int _max(lua_State *L)
{
	return reduce(opMax(), L);
}

static const struct luaL_Reg funcs[] = {
	{"add", add},
	{"add_mat_vect", add_mat_vect},
	{"cce", cce},
	{"div_mat_vect", div_mat_vect},
	{"exp", _exp},
	{"get_cols", get_cols},
	{"huber", huber},
	{"huber_deriv", huber_deriv},
	{"max", _max},
	{"mult_by_sigmoid_deriv", mult_by_sigmoid_deriv},
	{"mult_by_tanh_deriv", mult_by_tanh_deriv},
	{"mult_mat_vect", mult_mat_vect},
	{"set_cols", set_cols},
	{"shrink", shrink},
	{"sigmoid", sigmoid},
	{"sub_mat_vect", sub_mat_vect},
	{"sum", sum},
	{"tanh", tanh},

	{NULL, NULL}
};

extern "C" int luaopen_libjzt(lua_State *L) {
	luaL_openlib(L, "jzt", funcs, 0);
	return 1;
}
