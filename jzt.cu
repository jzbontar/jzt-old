extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"

#define TB 128

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

static const struct luaL_Reg funcs[] = {
	{"get_cols", get_cols},
	{"set_cols", set_cols},

	{NULL, NULL}
};

extern "C" int luaopen_libjzt(lua_State *L) {
	luaL_openlib(L, "jzt", funcs, 0);
	return 1;
}
