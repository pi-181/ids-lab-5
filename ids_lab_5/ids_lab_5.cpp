#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
	int rank, numprocs, R1, C1, R2, C2, count, remainder, myRowsSize;
	clock_t tStart = NULL;
	double execTime = 0;

	int* matrix1 = NULL;
	int* matrix2 = NULL;

	int* result = NULL;

	int* sendcounts = NULL;
	int* senddispls = NULL;

	int* recvcounts = NULL;
	int* recvdispls = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)
	{
		cout << "1s matrix rows:" << endl;
		cin >> R1;
		if (R1 < 1)	return EXIT_FAILURE;

		cout << "1st matrix columns:" << endl;
		cin >> C1;
		if (C1 < 1)	return EXIT_FAILURE;

		R2 = C1;
		cout << "2nd matrix columns:" << endl;
		cin >> C2;
		if (C2 < 1)	return EXIT_FAILURE;

		cout << endl;

		// генеруємо матриці
		matrix1 = new int[R1 * C1];
		cout << "matrix 1:" << endl;
		for (int r = 0; r < R1; ++r)
		{
			for (int c = 0; c < C1; ++c)
			{
				matrix1[C1 * r + c] = rand() % 100;
				cout << matrix1[C1 * r + c] << '\t';
			}
			cout << endl;
		}
		cout << endl;

		matrix2 = new int[R2 * C2];
		cout << "matrix 2:" << endl;
		for (int r = 0; r < R2; ++r)
		{
			for (int c = 0; c < C2; ++c)
			{
				matrix2[C2 * r + c] = rand() % 100;
				cout << matrix2[C2 * r + c] << '\t';
			}
			cout << endl;
		}
		cout << endl;


		sendcounts = new int[numprocs];
		senddispls = new int[numprocs];

		recvcounts = new int[numprocs];
		recvdispls = new int[numprocs];

		count = R1 / numprocs;
		remainder = R1 - count * numprocs;

		int prefixSum = 0;
		for (int i = 0; i < numprocs; ++i)
		{
			int t1 = (i < remainder) ? count + 1 : count;
			recvcounts[i] = t1 * C2;
			sendcounts[i] = t1 * C1;

			int t2 = prefixSum;
			recvdispls[i] = t2 * C2;
			senddispls[i] = t2 * C1;

			prefixSum += t1;
		}
	}

	if (rank == 0)
		tStart = clock();

	// Передаємо дані всій групі
	MPI_Bcast(&R1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&C1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&R2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&C2, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0) matrix2 = new int[R2 * C2];
	MPI_Bcast(matrix2, R2 * C2, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0)
	{
		count = R1 / numprocs;
		remainder = R1 - count * numprocs;
	}

	myRowsSize = rank < remainder ? count + 1 : count;
	cout << "Rank: " << rank << ",  myRowsSize: " << myRowsSize << endl;

	int* matrixPart = new int[myRowsSize * C1];

	// Розподіляємо дані по всім учасникам групи
	MPI_Scatterv(matrix1, sendcounts, senddispls, MPI_INT, matrixPart, myRowsSize * C1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		delete[] sendcounts;
		delete[] senddispls;
		delete[] matrix1;
	}

	int* resultPart = new int[myRowsSize * C2];

	// Вказуємо препроцесору що даний цикл потрібно розпаралелити по учасниках групи
	// (Інкремент і стартове значення зміниться на номер учасника)
	#pragma omp parallel for
	for (int i = 0; i < myRowsSize; ++i)
	{
		for (int j = 0; j < C2; ++j)
		{
			resultPart[i * C2 + j] = 0;
			for (int k = 0; k < C1; ++k)
			{
				resultPart[i * C2 + j] += matrixPart[i * C1 + k] * matrix2[k * C2 + j];
			}
		}
	}
	delete[] matrixPart;
	delete[] matrix2;

	if (rank == 0)
		result = new int[R1 * C2];

	// Збираємо дані від всіх
	MPI_Gatherv(resultPart, myRowsSize * C2, MPI_INT, result, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
		execTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;

	delete[] resultPart;

	if (rank == 0)
	{
		delete[] recvcounts;
		delete[] recvdispls;
	}

	// Завершуємо середу виконання процесу MPI.
	MPI_Finalize();

	if (rank == 0)
	{
		
		cout << "result:" << endl;
		for (int i = 0; i < R1; ++i)
		{
			for (int j = 0; j < C2; ++j)
				cout << result[i * C2 + j] << '\t';
			cout << endl;
		}
		
		delete[] result;

		printf("Time taken: %.2fs\n", execTime);
	}

	return EXIT_SUCCESS;
}