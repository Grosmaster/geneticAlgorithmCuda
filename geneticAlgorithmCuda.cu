#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "pch.h"
#include <iostream>
using namespace std;
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib> // Для работы с функцией system()
/*
const int sizePoint = 5;
const int sizeIndividum = 5;
const int mathValueMutation = 5;
const float dispersionMutation = 5.0f;
const int powCount = 3;
const float randMaxCount = 20.0f;
*/

const int sizePoint = 500;
const int sizeIndividum = 1000;
const int mathValueMutation = 5;
const float dispersionMutation = 5.0f;
const int powCount = 3;
const float randMaxCount = 20.0f;
const int maxPokoleney = 30;

__global__ void errorsKernel(float *points, float *individs, float *errors, int powCount, int sizePoint)
{

	int id = threadIdx.x;
	float ans = 0;
	int x = 1;
	for (int i = 0; i < sizePoint; i++)
	{
		for (int j = 0; j < powCount; j++)
		{
			for (int k = 0; k < j; k++)
			{
				x *= i;
			}
			x *= individs[id*powCount + j];
			ans += x;
			x = 1;
		}

		ans = points[i] - ans;
		errors[id] += sqrt(ans * ans);
		ans = 0;
	}
}


void testErrorsKernel(float* points, float* individs, float* errors, int powCount, int sizePoint, int sizeIndividum)
{
	for (int id = 0; id < sizeIndividum; id++)
	{
		float ans = 0.0f;
		errors[id] = 0.0f;
		int x = 0;
		for (int i = 0; i < sizePoint; i++)
		{
			for (int j = 0; j < powCount; j++)
			{
				x = pow(i, j);
				x *= individs[id*powCount + j];
				ans += x;
				x = 0;
			}

			ans = points[i] - ans;
			errors[id] += sqrt(ans * ans);
			ans = 0;
		}
	}
}

float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void cpu() {
	float* pointsH = new float[sizePoint];
	for (int i = 0; i < sizePoint; i++)
	{
		pointsH[i] = RandomFloat(0, 20);
	}

	float* individumsH = new float[sizeIndividum * powCount];
	for (int i = 0; i < sizeIndividum * powCount; i++)
	{
		individumsH[i] = RandomFloat(0, randMaxCount);
	}

	float* errorsH = new float[sizeIndividum];
	for (int i = 0; i < sizeIndividum; i++)
	{
		errorsH[i] = 1000;
	}

	unsigned int start_time = clock(); // начальное время

	for (int pokolenie = 0; pokolenie < maxPokoleney; pokolenie++)
	{
		testErrorsKernel(pointsH, individumsH, errorsH, powCount, sizePoint, sizeIndividum);

		float* errorsCrossOver = new float[sizeIndividum];

		for (size_t i = 0; i != sizeIndividum; ++i)
			errorsCrossOver[i] = errorsH[i];
		sort(errorsCrossOver, errorsCrossOver + sizeIndividum);

		int merodianCrossOvering = sizeIndividum / 2;
		float merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering];
		float* theBestInd = new float[powCount];

		for (size_t i = 0; i < sizeIndividum; i++)
		{
			if (merodianErrorCrossOvering < errorsH[i]) {
				for (size_t j = 0; j < powCount; j++)
				{
					individumsH[i * powCount + j] = 0;
				}
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < powCount; j++)
				{
					theBestInd[j] = individumsH[i *  powCount + j];
				}
			}
		}

		printf("error = %f\n", errorsCrossOver[0]);

		for (int i = 0; i < sizeIndividum * powCount; i++)
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % powCount];
			}

			if (mathValueMutation >(rand() % 100 + 1)) {
				individumsH[i] += RandomFloat(-dispersionMutation, dispersionMutation);
			}
		}
	}
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	printf("search_time_cpu = %i\n", search_time);
}

void gpu() {
	float* pointsH = new float[sizePoint];
	for (int i = 0; i < sizePoint; i++)
	{
		pointsH[i] = RandomFloat(0, 20);
	}

	float* individumsH = new float[sizeIndividum * powCount];
	for (int i = 0; i < sizeIndividum * powCount; i++)
	{
		individumsH[i] = RandomFloat(0, randMaxCount);
	}

	float* errorsH = new float[sizeIndividum];
	for (int i = 0; i < sizeIndividum; i++)
	{
		errorsH[i] = 1000;
	}

	unsigned int start_time_gpu = clock(); // начальное время
	float* pointsD = NULL;
	float* individumsD = NULL;
	float* errorsD = NULL;

	for (int pokolenie = 0; pokolenie < maxPokoleney; pokolenie++)
	{

		int sizeIndividumBytes = sizeIndividum * powCount * sizeof(float);
		int sizePointBytes = sizePoint * sizeof(float);

		cudaMalloc((void**)&pointsD, sizePointBytes);
		cudaMalloc((void**)&individumsD, sizeIndividumBytes*powCount);
		cudaMalloc((void**)&errorsD, sizeIndividum * sizeof(float));

		cudaMemcpy(pointsD, pointsH, sizePointBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(individumsD, individumsH, sizeIndividumBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(errorsD, errorsH, sizeIndividumBytes, cudaMemcpyHostToDevice);

		errorsKernel << <1, sizeIndividum >> > (pointsD, individumsD, errorsD, powCount, sizePoint);

		cudaMemcpy(errorsH, errorsD, sizeIndividum * sizeof(float), cudaMemcpyDeviceToHost);

		//----------------------
		float* errorsCrossOver = new float[sizeIndividum];

		for (size_t i = 0; i != sizeIndividum; ++i)
			errorsCrossOver[i] = errorsH[i];
		sort(errorsCrossOver, errorsCrossOver + sizeIndividum);
		printf("error = %f\n", errorsCrossOver[0]);
		int merodianCrossOvering = sizeIndividum / 2;
		float merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering];
		float* theBestInd = new float[powCount];

		for (size_t i = 0; i < sizeIndividum; i++)
		{
			if (merodianErrorCrossOvering < errorsH[i]) {
				for (size_t j = 0; j < powCount; j++)
				{
					individumsH[i * powCount + j] = 0;
				}
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < powCount; j++)
				{
					theBestInd[j] = individumsH[i *  powCount + j];
				}
			}
		}

		for (int i = 0; i < sizeIndividum * powCount; i++)
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % powCount];
			}

			if (mathValueMutation >(rand() % 100 + 1)) {
				individumsH[i] += RandomFloat(-dispersionMutation, dispersionMutation);
			}
		}
	}
	unsigned int end_time_gpu = clock(); // конечное время
	unsigned int search_time_gpu = end_time_gpu - start_time_gpu; // искомое время

	printf("search_time_gpu = %i\n", search_time_gpu);

	cudaFree(pointsD);
	cudaFree(individumsD);
	cudaFree(errorsD);

	delete pointsH;
	delete individumsH;
	delete errorsH;
}

int main()
{
	cpu();
	gpu();
	system("pause");
	return 0;
}
