#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif


__global__ void executeFirstLayer(float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU)
{
        __shared__ float max[512];
	int blockID=blockIdx.x;
	int pixelX=threadIdx.x;
	int pixelY=threadIdx.y;
	

    int kernelTemplate[25] = {
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };


	int weightBegin=blockID*26;
	int windowX=pixelX*2;
	int windowY=pixelY*2;
	max[windowX] = Layer1_Weights_GPU[windowX];
	__syncthreads();

	float result=0;

	result+=Layer1_Weights_GPU[weightBegin];

	++weightBegin;

	for(int i=0;i<25;++i)
	{
		result+=Layer1_Neurons_GPU[windowY*29+windowX+kernelTemplate[i]]*Layer1_Weights_GPU[weightBegin+i];
	}

	result=(1.7159*tanhf(0.66666667*result));

	Layer2_Neurons_GPU[13*13*blockID+pixelY*13+pixelX]=result;

}

__global__ void executeSecondLayer(float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU)
{
        __shared__ float max[512];
	int blockID=blockIdx.x;
	int pixelX=threadIdx.x;
	int pixelY=threadIdx.y;

	int kernelTemplate2[25] = {
        0,  1,  2,  3,  4,
        13, 14, 15, 16, 17, 
        26, 27, 28, 29, 30,
        39, 40, 41, 42, 43, 
        52, 53, 54, 55, 56   };

	int weightBegin=blockID*26*6;
	int windowX=pixelX*2;
	int windowY=pixelY*2;
	max[windowX] = Layer2_Weights_GPU[windowX];
	__syncthreads();
    
	float result=0;

	
	result+=Layer2_Weights_GPU[weightBegin];
	
	if(blockID==1 && pixelX==0 && pixelY==0)
	{
		result+=0;
	}

	++weightBegin;

	for (int i=0; i<25; ++i )
    {
        result+=Layer2_Neurons_GPU[windowX + 13*windowY +kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6];
        result+=Layer2_Neurons_GPU[169 + windowX + 13*windowY +kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6+1];
	result+=Layer2_Neurons_GPU[338 + windowX + 13*windowY + kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6+2];
        result+=Layer2_Neurons_GPU[507 + windowX + 13*windowY + kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6+3];
        result+=Layer2_Neurons_GPU[676 + windowX + 13*windowY + kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6+4];
        result+=Layer2_Neurons_GPU[845 + windowX + 13*windowY + kernelTemplate2[i]]*Layer2_Weights_GPU[weightBegin+i*6+5];
	}

	result=(1.7159*tanhf(0.66666667*result));

	Layer3_Neurons_GPU[5*5*blockID+pixelY*5+pixelX]=result;
}

__global__ void executeThirdLayer(float *Layer3_Neurons_GPU, float *Layer3_Weights_GPU,float *Layer4_Neurons_GPU)
{
        __shared__ float max[512];
	int blockID=blockIdx.x;
	int pixelY=threadIdx.y;


	int weightBegin=blockID*1251;
	max[weightBegin] = Layer3_Weights_GPU[weightBegin];
        __syncthreads();

	float result=0;

	result+=Layer3_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<1250; ++i )
    {
		result+=Layer3_Neurons_GPU[i]*Layer3_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer4_Neurons_GPU[blockID]=result;

}

__global__ void executeFourthLayer(float *Layer4_Neurons_GPU,float *Layer4_Weights_GPU,float *Layer5_Neurons_GPU)
{
        __shared__ float max[512];
	int blockID=blockIdx.x;
	int pixelY=threadIdx.y;


	int weightBegin=blockID*101;
	max[weightBegin] = Layer4_Weights_GPU[weightBegin];
        __syncthreads();
 
	float result=0;

	result+=Layer4_Weights_GPU[weightBegin];

	++weightBegin;

    for (int i=0; i<100; ++i )
    {
		result+=Layer4_Neurons_GPU[i]*Layer4_Weights_GPU[weightBegin+i];
    }

	result=(1.7159*tanhf(0.66666667*result));

	Layer5_Neurons_GPU[blockID]=result;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
