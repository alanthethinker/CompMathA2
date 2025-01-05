package main

import (
	"fmt"
	"math"
)

type Matrix struct {
	data [][]float64
	n    int
}

type Vector struct {
	data []float64
	n    int
}

func NewMatrix(data [][]float64) *Matrix {
	n := len(data)
	return &Matrix{data: data, n: n}
}

func NewVector(data []float64) *Vector {
	return &Vector{data: data, n: len(data)}
}

func CramerSolve(A *Matrix, b *Vector) *Vector {
	n := A.n
	result := make([]float64, n)
	detA := determinant(A.data)

	for i := 0; i < n; i++ {
		Ai := copyMatrix(A.data)
		for j := 0; j < n; j++ {
			Ai[j][i] = b.data[j]
		}
		result[i] = determinant(Ai) / detA
	}

	return NewVector(result)
}

func GaussSolve(A *Matrix, b *Vector) *Vector {
	n := A.n
	augmented := make([][]float64, n)
	for i := 0; i < n; i++ {
		augmented[i] = make([]float64, n+1)
		copy(augmented[i], A.data[i])
		augmented[i][n] = b.data[i]
	}

	for i := 0; i < n; i++ {
		pivot := augmented[i][i]
		for j := i + 1; j < n; j++ {
			factor := augmented[j][i] / pivot
			for k := i; k <= n; k++ {
				augmented[j][k] -= factor * augmented[i][k]
			}
		}
	}

	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := augmented[i][n]
		for j := i + 1; j < n; j++ {
			sum -= augmented[i][j] * x[j]
		}
		x[i] = sum / augmented[i][i]
	}

	return NewVector(x)
}

func JacobiSolve(A *Matrix, b *Vector, maxIter int, tolerance float64) *Vector {
	n := A.n
	x := make([]float64, n)
	xNew := make([]float64, n)

	for iter := 0; iter < maxIter; iter++ {
		for i := 0; i < n; i++ {
			sum := b.data[i]
			for j := 0; j < n; j++ {
				if j != i {
					sum -= A.data[i][j] * x[j]
				}
			}
			xNew[i] = sum / A.data[i][i]
		}

		maxDiff := 0.0
		for i := 0; i < n; i++ {
			diff := math.Abs(xNew[i] - x[i])
			if diff > maxDiff {
				maxDiff = diff
			}
			x[i] = xNew[i]
		}

		if maxDiff < tolerance {
			break
		}
	}

	return NewVector(x)
}

func GaussSeidel(A *Matrix, b *Vector, maxIter int, tolerance float64) *Vector {
	n := A.n
	x := make([]float64, n)

	for iter := 0; iter < maxIter; iter++ {
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			oldX := x[i]
			sum := b.data[i]
			for j := 0; j < n; j++ {
				if j != i {
					sum -= A.data[i][j] * x[j]
				}
			}
			x[i] = sum / A.data[i][i]
			diff := math.Abs(x[i] - oldX)
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		if maxDiff < tolerance {
			break
		}
	}

	return NewVector(x)
}

func determinant(matrix [][]float64) float64 {
	n := len(matrix)
	if n == 1 {
		return matrix[0][0]
	}
	if n == 2 {
		return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	}

	det := 0.0
	for i := 0; i < n; i++ {
		minor := make([][]float64, n-1)
		for j := range minor {
			minor[j] = make([]float64, n-1)
		}
		for j := 1; j < n; j++ {
			for k := 0; k < n; k++ {
				if k < i {
					minor[j-1][k] = matrix[j][k]
				} else if k > i {
					minor[j-1][k-1] = matrix[j][k]
				}
			}
		}
		if i%2 == 0 {
			det += matrix[0][i] * determinant(minor)
		} else {
			det -= matrix[0][i] * determinant(minor)
		}
	}
	return det
}

func copyMatrix(matrix [][]float64) [][]float64 {
	n := len(matrix)
	copy := make([][]float64, n)
	for i := range matrix {
		copy[i] = make([]float64, n)
		for j := range matrix[i] {
			copy[i][j] = matrix[i][j]
		}
	}
	return copy
}

func main() {
	A := NewMatrix([][]float64{
		{3, -5, 47, 20},
		{11, 16, 17, 10},
		{56, 22, 11, -18},
		{17, 66, -12, 7},
	})

	b := NewVector([]float64{18, 26, 34, 82})

	cramerSolution := CramerSolve(A, b)
	gaussSolution := GaussSolve(A, b)
	jacobiSolution := JacobiSolve(A, b, 1000, 1e-10)
	gaussSeidelSolution := GaussSeidel(A, b, 1000, 1e-10)

	fmt.Println("Cramer's Rule solution:")
	for i, val := range cramerSolution.data {
		fmt.Printf("x%d = %.6f\n", i+1, val)
	}

	fmt.Println("\nGauss Elimination solution:")
	for i, val := range gaussSolution.data {
		fmt.Printf("x%d = %.6f\n", i+1, val)
	}

	fmt.Println("\nJacobi Method solution:")
	for i, val := range jacobiSolution.data {
		fmt.Printf("x%d = %.6f\n", i+1, val)
	}

	fmt.Println("\nGauss-Seidel Method solution:")
	for i, val := range gaussSeidelSolution.data {
		fmt.Printf("x%d = %.6f\n", i+1, val)
	}
}
