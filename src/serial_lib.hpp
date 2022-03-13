#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000

using namespace std;

typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;


void init_matrix(Matrix &m, int nrow, int ncol) {
	m.row_eff = nrow;
	m.col_eff = ncol;

	for (int i = 0; i < m.row_eff; i++) {
		for (int j = 0; j < m.col_eff; j++) {
			m.mat[i][j] = 0;
		}
	}
}

Matrix input_matrix(ifstream &fs, int nrow, int ncol) {
	Matrix input;
	init_matrix(input, nrow, ncol);

	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
            fs >> input.mat[i][j];
		}
	}

	return input;
}

void print_matrix(Matrix &m) {
	for (int i = 0; i < m.row_eff; i++) {
		for (int j = 0; j < m.col_eff; j++) {
			printf("%d ", m.mat[i][j]);
		}
		printf("\n");
	}
}
