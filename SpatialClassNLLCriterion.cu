#include "luaT.h"
#include "THC.h"


#define SPATIALCLASSNLLCRITERION_THREADS 128
#define SPATIALCLASSNLLCRITERION_REDUCE 8
#define SPATIALCLASSNLLCRITERION_BLOCKS 32768


//Reducing inputs to buffer
__global__ void cunn_SpatialClassNLLCriterion_updateOutput_kernelReduceSumFromInput
(float *buffer, float *input, float *target, int target_size, int feature_size, int spatial_size, int reduce_size, int sizeAverage, int buffer_shift){
  int idx = blockIdx.x * blockDim.x + threadIdx.x + buffer_shift;
  float sum = 0.0;

  //Working with index of target
  int reduce_end = (idx + 1) * reduce_size;
  int input_index = 0;
	int tgt;
  for(int i = idx * reduce_size; i < target_size && i < reduce_end; ++i){
		tgt = ((int)target[i] - 1);
		// do not update negative targets (convention for no training)
		if (tgt >= 0) {
			input_index = i / spatial_size * feature_size + tgt * spatial_size + i % spatial_size;
			sum -= input[input_index];
		}
  }
  if (sizeAverage){
    sum /= (float)target_size;
  }

  //Reduce to buffer
  if (idx * reduce_size < target_size){
    buffer[idx] = sum;
  }
}

//Reducing buffer in a single kernel
__global__ void cunn_SpatialClassNLLCriterion_updateOutput_kernelReduceSum
(float *buffer, int buffer_size, int reduce_size, int buffer_step){
  __shared__ float tbuf[SPATIALCLASSNLLCRITERION_THREADS];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0;
  int reduce_step,i;

  //Fetch data to shared buffer
  int buffer_start = idx * reduce_size * buffer_step;
  int buffer_end = (idx + 1) * reduce_size * buffer_step;
  for(i = buffer_start; i < buffer_size && i < buffer_end; i += buffer_step){
    sum += buffer[i];
  }
  tbuf[threadIdx.x] = sum;
  __syncthreads();

  //Reduce all the way down
  buffer_start = threadIdx.x * reduce_size;
  buffer_end = (threadIdx.x + 1) * reduce_size;
  for(reduce_step = 1; reduce_step < buffer_size; reduce_step *= reduce_size){
    //Reduce using local buffer
    if(buffer_start * reduce_step < SPATIALCLASSNLLCRITERION_THREADS){
      sum = 0.0;
      for(i = buffer_start; i * reduce_step < SPATIALCLASSNLLCRITERION_THREADS && i < buffer_end; ++i){
	sum += tbuf[i];
      }
    }
    tbuf[threadIdx.x] = sum;
    __syncthreads();
  }

  //Copy data back to buffer
  if (threadIdx.x == 0){
    //idx is actually the start of a buffer for a block.
    buffer[idx * reduce_size * buffer_step] = tbuf[0];
  }
}

__global__ void 
cunn_SpatialClassNLLCriterion_updateGradInput_kernel
(float* gradInput_data, float* target_data, int target_size, int spatial_size,
 int feature_size, int reduce_size, int target_shift, float grad){
  //This is target index
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * reduce_size + target_shift;
	int tgt;
  for(int i = 0; i < reduce_size && i < target_size - idx; ++i){
		tgt = ((int)target_data[idx + i]-1);
		// do not update negative targets (convention for no training)
		if (tgt >= 0) {
			int input_index = (idx + i) / spatial_size * feature_size + tgt * spatial_size
				+ (idx + i) % spatial_size;
			gradInput_data[input_index] = grad;
		}
  }
}

static int cunn_SpatialClassNLLCriterion_updateOutput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  input = THCudaTensor_newContiguous(input);
  float *input_data = THCudaTensor_data(input);
  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  target = THCudaTensor_newContiguous(target);
  float *target_data = THCudaTensor_data(target);
  
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  output = THCudaTensor_newContiguous(output);
  float *output_data = THCudaTensor_data(output);

  int nframe = 1, dim = 0, height = 1, width = 1;

  if(input->nDimension != target->nDimension){
    THError("Input and target must have the same dimension");
  }

  if (input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
    if (target->size[0] != 1){
      THError("Target size does not match with input");
    }
  } else if(input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
    if (target->size[0] != nframe || target->size[1] != 1){
      THError("Target size does not match with input");
    }
  } else if (input->nDimension == 3) {
    nframe = 1;
    dim = input->size[0];
    width = input->size[1];
    height = input->size[2];
    if (target->size[0] != 1 || target->size[1] != width || target->size[2] != height){
      THError("Target size does not match with input");
    }
  } else if (input->nDimension == 4) {
    nframe = input->size[0];
    dim = input->size[1];
    width = input->size[2];
    height = input->size[3];
    if (target->size[0]!=nframe || target->size[1] != 1 || target->size[2] != width || target->size[3] != height){
      THError("Target size does not match with input");
    }
  } else {
    THArgCheck(0, 2, "4d input maximum expected");
  }

  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  int reduce_size = SPATIALCLASSNLLCRITERION_REDUCE;
  int spatial_size = width * height;
  int feature_size = dim * spatial_size;
  int target_size = nframe * spatial_size;
  int buffer_size = target_size/reduce_size;
  if (target_size % reduce_size != 0){
    buffer_size = buffer_size + 1;
  }
  float *buffer;
  cudaError errcode = cudaMalloc((void**)&buffer, sizeof(float)*buffer_size);
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  //Setting up nthreads, blocks and number of calls
  int nthreads = SPATIALCLASSNLLCRITERION_THREADS;
  int ncalls = buffer_size / (SPATIALCLASSNLLCRITERION_BLOCKS * nthreads);
  if(ncalls == 0 || buffer_size % (SPATIALCLASSNLLCRITERION_BLOCKS * nthreads) != 0){
    ncalls += 1;
  }

  // Call reducing to buffer
  int nblocks = SPATIALCLASSNLLCRITERION_BLOCKS;
  int buffer_shift = 0;
  for(int call = 0; call < ncalls; ++call){
    buffer_shift = call * SPATIALCLASSNLLCRITERION_BLOCKS * nthreads;
    if (buffer_size - buffer_shift < SPATIALCLASSNLLCRITERION_BLOCKS * nthreads){
      nblocks = (buffer_size - buffer_shift) / nthreads;
      if (nblocks == 0 || (buffer_size - buffer_shift) % nthreads != 0){
	nblocks += 1;
      }
    }
    cunn_SpatialClassNLLCriterion_updateOutput_kernelReduceSumFromInput<<<nblocks,nthreads>>>
      (buffer, input_data, target_data, target_size, feature_size, spatial_size, reduce_size, sizeAverage, buffer_shift);
  }

  //Second and further reduce using buffer to buffer reduction
  nblocks = SPATIALCLASSNLLCRITERION_BLOCKS;
  for(int buffer_step = 1; buffer_step < buffer_size;
      (INT_MAX / nthreads / reduce_size >= buffer_step) ? buffer_step *= nthreads * reduce_size : buffer_step = buffer_size){
    ncalls = buffer_size / buffer_step / SPATIALCLASSNLLCRITERION_BLOCKS / nthreads / reduce_size;
    if(ncalls == 0 || buffer_size % (buffer_step * SPATIALCLASSNLLCRITERION_BLOCKS * nthreads * reduce_size) != 0){
      ncalls += 1;
    }
    for(int call = 0; call < ncalls; ++call){
      buffer_shift = call * buffer_step * SPATIALCLASSNLLCRITERION_BLOCKS * nthreads * reduce_size;
      if ((buffer_size - buffer_shift) / SPATIALCLASSNLLCRITERION_BLOCKS / nthreads / reduce_size / buffer_step == 0){
	nblocks = (buffer_size - buffer_shift) / buffer_step / nthreads / reduce_size;
	if (nblocks == 0 || (buffer_size - buffer_shift) % (buffer_step * nthreads * reduce_size) != 0){
	  nblocks += 1;
	}
      }
      cunn_SpatialClassNLLCriterion_updateOutput_kernelReduceSum<<<nblocks,nthreads>>>
	(buffer + buffer_shift, buffer_size - buffer_shift, reduce_size, buffer_step);
    }
  }

  // Any errors?
  errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  //Copy buffer[0] value to output
  errcode = cudaMemcpy(output_data,buffer,sizeof(float),cudaMemcpyDeviceToDevice);
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  //Free the buffer
  errcode = cudaFree(buffer);
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(output);
  THCudaTensor_free(target);
  THCudaTensor_free(input);
  
  return 1;
}

static int cunn_SpatialClassNLLCriterion_updateGradInput(lua_State *L) {
  
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  input = THCudaTensor_newContiguous(input);
  // float *input_data = THCudaTensor_data(input);
  
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  target = THCudaTensor_newContiguous(target);
  float *target_data = THCudaTensor_data(target);

  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  gradInput = THCudaTensor_newContiguous(gradInput);
  float *gradInput_data = THCudaTensor_data(gradInput);

  int nframe = 1, dim = 0, height = 1, width = 1;

  if(input->nDimension != target->nDimension){
    THError("Input and target must have the same dimension");
  }

  if (input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
    if (target->size[0] != 1){
      THError("Target size does not match with input");
    }
  } else if(input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
    if (target->size[0] != nframe || target->size[1] != 1){
      THError("Target size does not match with input");
    }
  } else if (input->nDimension == 3) {
    nframe = 1;
    dim = input->size[0];
    width = input->size[1];
    height = input->size[2];
    if (target->size[0] != 1 || target->size[1] != width || target->size[2] != height){
      THError("Target size does not match with input");
    }
  } else if (input->nDimension == 4) {
    nframe = input->size[0];
    dim = input->size[1];
    width = input->size[2];
    height = input->size[3];
    if (target->size[0]!=nframe || target->size[1] != 1 || target->size[2] != width || target->size[3] != height){
      THError("Target size does not match with input");
    }
  } else {
    THArgCheck(0, 2, "4d input maximum expected");
  }

  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  int reduce_size = SPATIALCLASSNLLCRITERION_REDUCE;
  int spatial_size = width * height;
  int feature_size = dim * spatial_size;
  int target_size = nframe * spatial_size;

  float grad = -1.0;
  if(sizeAverage) {
    grad /= (float)target_size;
  }
  int nblocks = SPATIALCLASSNLLCRITERION_BLOCKS;
  int nthreads = SPATIALCLASSNLLCRITERION_THREADS;
  int ncalls = target_size / nblocks / nthreads / reduce_size;
  int target_shift = 0;
  if (ncalls == 0 || target_size % (SPATIALCLASSNLLCRITERION_BLOCKS * nthreads * reduce_size) != 0){
    ncalls += 1;
  }
  for(int call = 0; call < ncalls; ++call){
    target_shift = call * SPATIALCLASSNLLCRITERION_BLOCKS * nthreads * reduce_size;
    if(target_size - target_shift < SPATIALCLASSNLLCRITERION_BLOCKS * nthreads * reduce_size){
      nblocks = (target_size - target_shift) / nthreads / reduce_size;
      if(nblocks == 0 || nblocks % (nthreads * reduce_size) != 0){
	nblocks += 1;
      }
    }
    cunn_SpatialClassNLLCriterion_updateGradInput_kernel<<<nblocks,nthreads>>>
      (gradInput_data, target_data, target_size, spatial_size, feature_size, reduce_size, target_shift, grad);
  }

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(gradInput);
  THCudaTensor_free(target);
  THCudaTensor_free(input);
  
  return 1;
}


static const struct luaL_Reg cunn_SpatialClassNLLCriterion__ [] = {
  {"SpatialClassNLLCriterion_updateOutput", cunn_SpatialClassNLLCriterion_updateOutput},
  {"SpatialClassNLLCriterion_updateGradInput", cunn_SpatialClassNLLCriterion_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialClassNLLCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialClassNLLCriterion__, "nn");
  lua_pop(L,1);
}
