// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <NN_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	NeuralNetwork();
}

void InitGPUMem(float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,float *Layer3_Weights_GPU,float *Layer4_Neurons_GPU,float *Layer4_Weights_GPU,float *Layer5_Neurons_GPU)
{
	cudaMalloc((void**) &Layer1_Neurons_GPU, sizeof(float)*29*29);
	cudaMalloc((void**) &Layer1_Weights_GPU, sizeof(float)*156);
	
	cudaMalloc((void**) &Layer2_Neurons_GPU, sizeof(float)*13*13*6);
	cudaMalloc((void**) &Layer2_Weights_GPU, sizeof(float)*7800);

	cudaMalloc((void**) &Layer3_Neurons_GPU, sizeof(float)*1250);
	cudaMalloc((void**) &Layer3_Weights_GPU, sizeof(float)*125100);

	cudaMalloc((void**) &Layer4_Neurons_GPU, sizeof(float)*100);
	cudaMalloc((void**) &Layer4_Weights_GPU, sizeof(float)*1010);

	cudaMalloc((void**) &Layer5_Neurons_GPU, sizeof(float)*10);
}
void InitHostMem(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU)
{
	// initial layer 1 weight
	FILE * pFile1 = fopen ("lw1.wei","rb");
	if (pFile1 != NULL)
	{
	for(int i=0;i<156;++i)
		fread(&(Layer1_Weights_CPU[i]),sizeof(float),1,pFile1);
		fclose (pFile1);
	}

	// initial layer 2 weight
	FILE * pFile2 = fopen ("lw2.wei","rb");
	if (pFile2 != NULL)
	{
		fread(Layer2_Weights_CPU,sizeof(float),7800,pFile2);
		fclose (pFile2);
	}
	// initial layer 3 weight
	FILE * pFile3 = fopen ("lw3.wei","rb");
	if (pFile3 != NULL)
	{
		fread(Layer3_Weights_CPU,sizeof(float),125100,pFile3);
		fclose (pFile3);
	}
	// initial layer 4 weight
	FILE * pFile4 = fopen ("lw4.wei","rb");
	if (pFile4 != NULL)
	{
		fread(Layer4_Weights_CPU,sizeof(float),1010,pFile4);
		fclose (pFile4);
	}
}

void readIn(float *layer1)
{
	FILE *fp;
	fp=fopen("in.neu","rb");
	if(fp)
	{
		fread(layer1,sizeof(float),29*29,fp);
		fclose(fp);
	}
}

void output(double *final)
{
	FILE *fp=0;
	fp=fopen("out.res","wb");
	if(fp)
	{
		fwrite(final,sizeof(double),10,fp);
		fclose(fp);
	}
}

void NeuralNetwork()
{
	float Layer1_Neurons_CPU[29*29]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

//readIn(Layer1_Neurons_CPU);

	float *Layer1_Neurons_GPU;
	float Layer1_Weights_CPU[156];
	float *Layer1_Weights_GPU;

	float Layer2_Weights_CPU[7800];
	float *Layer2_Weights_GPU;
	float *Layer2_Neurons_GPU;

	float Layer3_Weights_CPU[125100];
	float *Layer3_Weights_GPU;
	float *Layer3_Neurons_GPU;

	float Layer4_Weights_CPU[1010];
	float *Layer4_Weights_GPU;
	float *Layer4_Neurons_GPU;

	float Layer5_Neurons_CPU[10]={0,0,0,0,0,0,0,0,0,0};
	float *Layer5_Neurons_GPU;

	double *outputLayer;
	unsigned int timer = 0;
	float totaltime = 0.0f;
	//init input here
	InitHostMem(Layer1_Weights_CPU,Layer2_Weights_CPU,Layer3_Weights_CPU,Layer4_Weights_CPU);


	//allocate momory on Device
	//InitGPUMem(Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer2_Neurons_GPU,Layer2_Weights_GPU,Layer3_Neurons_GPU,Layer3_Weights_GPU,Layer4_Neurons_GPU,Layer4_Weights_GPU,Layer5_Neurons_GPU);
	cudaMalloc((void**) &Layer1_Neurons_GPU, sizeof(float)*29*29);
	cudaMalloc((void**) &Layer1_Weights_GPU, sizeof(float)*156);
	
	cudaMalloc((void**) &Layer2_Neurons_GPU, sizeof(float)*13*13*6);
	cudaMalloc((void**) &Layer2_Weights_GPU, sizeof(float)*7800);

	cudaMalloc((void**) &Layer3_Neurons_GPU, sizeof(float)*1250);
	cudaMalloc((void**) &Layer3_Weights_GPU, sizeof(float)*125100);

	cudaMalloc((void**) &Layer4_Neurons_GPU, sizeof(float)*100);
	cudaMalloc((void**) &Layer4_Weights_GPU, sizeof(float)*1010);

	cudaMalloc((void**) &Layer5_Neurons_GPU, sizeof(float)*10);
	outputLayer = (double*)malloc(sizeof(double)*10);
	//init 29x29 handwritting array
	// already done in "initial"

	//copy from CPU to GPU
	cudaMemcpy(Layer1_Neurons_GPU,Layer1_Neurons_CPU, sizeof(float)*29*29, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Weights_GPU,Layer1_Weights_CPU, sizeof(float)*156, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer2_Weights_GPU,Layer2_Weights_CPU, sizeof(float)*7800, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer3_Weights_GPU,Layer3_Weights_CPU, sizeof(float)*125100, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer4_Weights_GPU,Layer4_Weights_CPU, sizeof(float)*1010, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer5_Neurons_GPU,Layer5_Neurons_CPU, sizeof(float)*10, cudaMemcpyHostToDevice);

	// CUT_SAFE_CALL(cutCreateTimer(&timer));
	// CUT_SAFE_CALL(cutStartTimer(timer));

	dim3 Layer1_Block(6,1);
	dim3 Layer1_Thread(13,13);
	executeFirstLayer<<<Layer1_Block,Layer1_Thread>>>(Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer2_Neurons_GPU);

	dim3 Layer2_Block(50,1);
	dim3 Layer2_Thread(5,5);
	executeSecondLayer<<<Layer2_Block,Layer2_Thread>>>(Layer2_Neurons_GPU, Layer2_Weights_GPU,Layer3_Neurons_GPU);

	dim3 Layer3_Block(100,1);
	dim3 Layer3_Thread(1,1);
	executeThirdLayer<<<Layer3_Block,Layer3_Thread>>>(Layer3_Neurons_GPU, Layer3_Weights_GPU,Layer4_Neurons_GPU);

	dim3 Layer4_Block(10,1);
	dim3 Layer4_Thread(1,1);
	executeFourthLayer<<<Layer4_Block,Layer4_Thread>>>(Layer4_Neurons_GPU,Layer4_Weights_GPU,Layer5_Neurons_GPU);

	


//	totaltime = cutGetTimerValue(timer);

	//copy from GPU to CPU
   cudaMemcpy(Layer5_Neurons_CPU,Layer5_Neurons_GPU, sizeof(float)*10, cudaMemcpyDeviceToHost);
    
    // stop and destroy timer

    //printf("Processing time: %f (ms) \n", totaltime);
	//  CUT_SAFE_CALL(cutDeleteTimer(timer));

	for(int a=0;a<10;a++)
	{
		outputLayer[a] = (double)Layer5_Neurons_CPU[a];
	}
	output(outputLayer);


	float Layer4_Neurons_CPU[100];
	cudaMemcpy(Layer4_Neurons_CPU,Layer4_Neurons_GPU,sizeof(float)*100,cudaMemcpyDeviceToHost);
	FILE *fp=fopen("layer_4.neu","wb");
	fwrite(Layer4_Neurons_CPU,sizeof(float),100,fp);
	fclose(fp);

	float Layer3_Neurons_CPU[50*5*5];
	cudaMemcpy(Layer3_Neurons_CPU,Layer3_Neurons_GPU,sizeof(float)*50*5*5,cudaMemcpyDeviceToHost);
	fp=fopen("layer_3.neu","wb");
	fwrite(Layer3_Neurons_CPU,sizeof(float),50*5*5,fp);
	fclose(fp);

	float Layer2_Neurons_CPU[13*13*6];
	cudaMemcpy(Layer2_Neurons_CPU,Layer2_Neurons_GPU,sizeof(float)*13*13*6,cudaMemcpyDeviceToHost);
	fp=fopen("layer_2.neu","wb");
	fwrite(Layer2_Neurons_CPU,sizeof(float),13*13*6,fp);
	fclose(fp);

	fp=fopen("layer_1.neu","wb");
	fwrite(Layer1_Neurons_CPU,sizeof(float),29*29,fp);
	fclose(fp);

	exit(0);
}