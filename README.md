# Genetic Algorithm Cuda

Аппроксимация функции с помощью генетического алгоритма

Задача аппроксимации - есть множество точек, принадлежащих графику некой функции. Нужно найти полином N-ой степени, проходящий как можно ближе к этим точкам.

На вход алгоритму падается:
const int sizePoint = 500; //Количество точек
const int sizeIndividum = 1000; //Количество индивидумов
const int mathValueMutation = 5; //Вероятность мутации в процентах 
const float dispersionMutation = 5.0f; //Разброс мутации
const int powCount = 3; //Стемень полинома
const float randMaxCount = 20.0f; //Верхняя граница для заполнения точек
const int maxPokoleney = 30; //Максимальное количесто поколений

Алгоритм:

Создание случайного набора точек

	for (int i = 0; i < sizePoint; i++)
	{
		pointsH[i] = RandomFloat(0, randMaxCount);
	}

Создание парвого поколения

	for (int i = 0; i < sizeIndividum * powCount; i++)
	{
		individumsH[i] = RandomFloat(0, randMaxCount);
	}

Вычисление ошибки

cpu

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

gpu

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

Смена поколений:

Худшая половина зануляется. Нулевые хромосомы заполяняются рандомно хромосомами лучшей особи. Для каждой хромосомы возможно мутация с заданой вероятностью и мерой разброса.

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

Для видео карты GeForce GTX 950 ускорение составило 7 раз.

![photo_2019-12-29_02-35-19.png](\photo_2019-12-29_02-35-19.png)
