#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))


namespace vllm {
	
}