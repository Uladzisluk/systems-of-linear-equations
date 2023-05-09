#include <iostream>
#include <cmath>
#include <chrono>
#include "matplotlibcpp.h"

using namespace std;

namespace plt = matplotlibcpp;

chrono::steady_clock sc;

typedef struct Parameters {
	int iterations;
	double time;
	double norm_error;
}Parameters;

typedef struct Matrix {
	double** A;
	int size;
}Matrix;

Matrix* copy_matrix(Matrix* m) {
	int N = m->size;
	Matrix* matrix;
	matrix = (Matrix*)malloc(sizeof(Matrix));
	matrix->size = N;
	matrix->A = (double**)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) {
		matrix->A[i] = (double*)malloc(sizeof(double) * N);
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix->A[i][j] = m->A[i][j];
		}
	}
	return matrix;
}

Matrix* create_matrix_A(int a1, int a2, int a3, int N) {
	Matrix* matrix;
	matrix = (Matrix*)malloc(sizeof(Matrix));
	matrix->size = N;
	matrix->A = (double**)malloc(sizeof(double*)*N);
	for (int i = 0; i < N; i++) {
		matrix->A[i] = (double*)malloc(sizeof(double) * N);
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) matrix->A[i][j] = a1;
			else if (j == i + 1 || j == i - 1) matrix->A[i][j] = a2;
			else if (j == i + 2 || j == i - 2) matrix->A[i][j] = a3;
			else matrix->A[i][j] = 0;
		}
	}
	return matrix;
}

void residual(Matrix* A, double* b, double* x, double* res) {
	for (int i = 0; i < A->size; i++)
	{
		double sum = 0.0;
		for (int j = 0; j < A->size; j++)
		{
			sum += A->A[i][j] * x[j];
		}
		res[i] = sum - b[i];
	}
}

double norm(double* res, int N) {
	double n = 0;
	for (int i = 0; i < N; i++) {
		n += pow(res[i], 2);
	}
	return sqrt(n);
}

Parameters jacobi(Matrix* A, double* b, double* x, double norma) {
	int N = A->size;
	Parameters parameters;
	parameters.iterations = 0;
	parameters.time = 0.0;
	double prev_norm;
	double* res = (double*)malloc(sizeof(double) * N);
	double* prev_x = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) prev_x[i] = 1;
	residual(A, b, x, res);
	auto start = sc.now();

	while ((prev_norm=norm(res, N)) > norma) {
		for (int i = 0; i < N; i++) {
			double sum = 0.0;
			for (int j = 0; j < N; j++) {
				if (i != j) {
					sum += A->A[i][j] * prev_x[j];
				}
			}
			x[i] = (b[i] - sum) / A->A[i][i];
		}

		for (int i = 0; i < N; i++) prev_x[i] = x[i];
		parameters.iterations++;
		auto time = static_cast<chrono::duration<double>>(sc.now() - start);
		if (parameters.iterations % 100 == 0) {
			cout << "The algorithm has already been running for " << time.count() << " seconds" << endl;;
			cout << "The residual error norm equals " << norm(res, N) << endl;
		}
		residual(A, b, x, res);
		if (norm(res, N) >= INFINITY) {
			parameters.time = time.count();
			cout << "The residual error norm for the Jacobi method increases with each iteration." << endl;
			parameters.norm_error = norm(res, N);
			return parameters;
		}
	}
	auto end = sc.now();
	auto time_span = static_cast<chrono::duration<double>>(end - start);
	parameters.time = time_span.count();
	parameters.norm_error = norm(res, N);
	return parameters;
}

Parameters gauss_seidel(Matrix* A, double* b, double* x, double norma) {
	int N = A->size;
	Parameters parameters;
	parameters.iterations = 0;
	parameters.time = 0.0;
	double* res = (double*)malloc(sizeof(double) * N);
	double* y = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) x[i] = 1;
	residual(A, b, x, res);
	double prev_norm;
	auto start = sc.now();

	while ((prev_norm = norm(res, N)) > norma) {
		for (int i = 0; i < N; i++) {
			double sum = 0.0;
			for (int j = 0; j < N; j++) {
				if (j != i) {
					sum += A->A[i][j] * x[j];
				}
			}
			x[i] = (b[i] - sum) / A->A[i][i];
		}
		parameters.iterations++;
		residual(A, b, x, res);
		auto time = static_cast<chrono::duration<double>>(sc.now() - start);
		if (parameters.iterations % 100 == 0) {
			cout << "The algorithm has already been running for " << time.count() << " seconds" << endl;;
			cout << "The residual error norm equals " << norm(res, N) << endl;
		}
		if (norm(res, N) >= INFINITY) {
			parameters.time = time.count();
			cout << "The residual error norm for the Gauss-Seidel method increases with each iteration." << endl;
			parameters.norm_error = norm(res, N);
			return parameters;
		}
	}

	auto end = sc.now();
	auto time_span = static_cast<chrono::duration<double>>(end - start);
	parameters.time = time_span.count();
	parameters.norm_error = norm(res, N);
	return parameters;
}

Parameters LU_factorization(Matrix* A, double* b, double* x) {
	int N = A->size;
	double* res = (double*)malloc(sizeof(double) * N);
	Matrix* M = copy_matrix(A);
	double eps = 1e-12;
	
	auto start = sc.now();
	for (int k = 0; k < N - 1; k++) {
		if (fabs(M->A[k][k]) < eps) break;;
		for (int i = k + 1; i < N; i++) {
			M->A[i][k] /= M->A[k][k];
		}
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				M->A[i][j] -= M->A[i][k] * M->A[k][j];
			}
		}
	}

	Matrix* L = create_matrix_A(1, 0, 0, N);
	Matrix* U = create_matrix_A(0, 0, 0, N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (j >= i) U->A[i][j] = M->A[i][j];
			else if (j < i) L->A[i][j] = M->A[i][j];
		}
	}

	double* y = new double[N];

	for (int i = 0; i < N; ++i)
	{
		double S = 0;

		for (int j = 0; j < i; ++j) S += L->A[i][j] * y[j];

		y[i] = (b[i] - S) / L->A[i][i];
	}

	for (int i = N - 1; i >= 0; --i)
	{
		double S = 0;

		for (int j = i + 1; j < N; ++j) S += U->A[i][j] * x[j];

		x[i] = (y[i] - S) / U->A[i][i];
	}

	auto end = sc.now();
	auto time_span = static_cast<chrono::duration<double>>(end - start);
	Parameters parameters;
	parameters.iterations = 1;
	parameters.time = time_span.count();
	residual(A, b, x, res);
	parameters.norm_error = norm(res, N);


	return parameters;
}

void clearMatrix(Matrix* matrix) {
	for (int i = 0; i < matrix->size; i++) {
		free(matrix->A[i]);
	}
}

void zadanieA() {
	int a1 = 5 + 6;
	int a2 = -1, a3 = -1;
	int N = 908, f = 1;

	Matrix* A = create_matrix_A(a1, a2, a3, N);
	double* b = (double*)malloc(sizeof(double) * N);
	double* x = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0;
		b[i] = sin(i * (f + 1));
	}
}

void zadanieB() {
	int a1 = 5 + 6;
	int a2 = -1, a3 = -1;
	int N = 908, f = 1;

	Matrix* A = create_matrix_A(a1, a2, a3, N);
	double* b = (double*)malloc(sizeof(double) * N);
	double* x = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0;
		b[i] = sin(i * (f + 1));
	}

	cout << "Jacobi method started" << endl;
	Parameters parameters = jacobi(A, b, x, 1e-9);
	cout << "The residual error norm equals: " << parameters.norm_error << endl;
	cout << "Jacobi method took: " << parameters.time << " seconds." << endl;
	cout << "Number of iterations for Jacobi method: " << parameters.iterations << endl;
	cout << endl << "Gauss-Seidel method started" << endl;
	parameters = gauss_seidel(A, b, x, 1e-9);
	cout << "The residual error norm equals: " << parameters.norm_error << endl;
	cout << "Gauss-Seidel took: " << parameters.time << " seconds." << endl;
	cout << "Number of iterations for Gauss-Seidel method: " << parameters.iterations << endl;
}

void zadanieC() {
	int a1 = 3;
	int a2 = -1, a3 = -1;
	int N = 908, f = 1;

	Matrix* A = create_matrix_A(a1, a2, a3, N);
	double* b = (double*)malloc(sizeof(double) * N);
	double* x = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0;
		b[i] = sin(i * (f + 1));
	}


	cout << "Jacobi method started" << endl;
	Parameters parameters = jacobi(A, b, x, 1e-9);
	cout << "The residual error norm equals: " << parameters.norm_error << endl;
	cout << "Jacobi method took: " << parameters.time << " seconds." << endl;
	cout << "Number of iterations for Jacobi method: " << parameters.iterations << endl;
	cout << endl << "Gauss-Seidel method started" << endl;
	parameters = gauss_seidel(A, b, x, 1e-9);
	cout << "The residual error norm equals: " << parameters.norm_error << endl;
	cout << "Gauss-Seidel took: " << parameters.time << " seconds." << endl;
	cout << "Number of iterations for Gauss-Seidel method: " << parameters.iterations << endl;
}

void zadanieD() {
	int a1 = 3;
	int a2 = -1, a3 = -1;
	int N = 908, f = 1;

	Matrix* A = create_matrix_A(a1, a2, a3, N);
	double* b = (double*)malloc(sizeof(double) * N);
	double* x = (double*)malloc(sizeof(double) * N);
	for (int i = 0; i < N; i++) {
		x[i] = 1.0;
		b[i] = sin(i * (f + 1));
	}

	cout << "LU factorization started" << endl;
	Parameters parameters =  LU_factorization(A, b, x);
	cout << "LU factorization took: " << parameters.time << " seconds." << endl;
	cout << "The residual error norm equals: " << parameters.norm_error << endl;
}

void zadanieE() {
	int a1 = 5 + 6;
	int a2 = -1, a3 = -1;
	int f = 1;


	vector<int> N = {100, 500, 1000, 2000, 3000, 4000, 5000, 6000};
	int size = sizeof(N)/sizeof(int);
	cout << size << endl;
	Parameters* parameters = (Parameters*)malloc(sizeof(Parameters) * size);
	vector<int> iterations(size);
	vector<double> times(size);
	vector<double> error(size);

	cout << "For Jacobi:" << endl;

	for (int i = 0; i < size; i++) {
		Matrix* A = create_matrix_A(a1, a2, a3, N[i]);
		double* b = (double*)malloc(sizeof(double) * N[i]);
		double* x = (double*)malloc(sizeof(double) * N[i]);
		for (int j = 0; j < N[i]; j++) {
			x[j] = 1.0;
			b[j] = sin(j * (f + 1));
		}

		parameters[i] = jacobi(A, b, x, 1e-9);
		iterations.at(i) = parameters[i].iterations;
		times.at(i) = parameters[i].time;
		error.at(i) = parameters[i].norm_error;
		free(b);
		free(x);
		clearMatrix(A);
		free(A);
		cout << "N: " << N[i] << " Iterations: " << iterations.at(i) << " Time: " << times.at(i);
		cout << " Error: " << error.at(i) << endl;
	}

	cout << endl << "For Gauss-Seidel:" << endl;
	for (int i = 0; i < size; i++) {
		Matrix* A = create_matrix_A(a1, a2, a3, N[i]);
		double* b = (double*)malloc(sizeof(double) * N[i]);
		double* x = (double*)malloc(sizeof(double) * N[i]);
		for (int j = 0; j < N[i]; j++) {
			x[j] = 1.0;
			b[j] = sin(j * (f + 1));
		}

		parameters[i] = gauss_seidel(A, b, x, 1e-9);
		iterations.at(i) = parameters[i].iterations;
		times.at(i) = parameters[i].time;
		error.at(i) = parameters[i].norm_error;
		free(b);
		free(x);
		clearMatrix(A);
		free(A);
		cout << "N: " << N[i] << " Iterations: " << iterations.at(i) << " Time: " << times.at(i);
		cout << " Error: " << error.at(i) << endl;
	}
		
	cout << endl << "For LU factorization:" << endl;
	for (int i = 0; i < size; i++) {
		Matrix* A = create_matrix_A(a1, a2, a3, N[i]);
		double* b = (double*)malloc(sizeof(double) * N[i]);
		double* x = (double*)malloc(sizeof(double) * N[i]);
		for (int j = 0; j < N[i]; j++) {
			x[j] = 1.0;
			b[j] = sin(j * (f + 1));
		}

		parameters[i] = LU_factorization(A, b, x);
		iterations.at(i) = parameters[i].iterations;
		times.at(i) = parameters[i].time;
		error.at(i) = parameters[i].norm_error;
		free(b);
		free(x);
		clearMatrix(A);
		free(A);
		cout << "N: " << N[i] << " Iterations: " << iterations.at(i) << " Time: " << times.at(i);
		cout << " Error: " << error.at(i) << endl;
	}
}

int main() {
	cout << "Zadanie B:" << endl;
	zadanieB();
	cout << endl << "Zadanie C:" << endl;
	zadanieC();
	cout << endl << "Zadanie D:" << endl;
	zadanieD();
	//cout << endl << "Zadanie E:" << endl;
	//zadanieE();

	return 0;
}