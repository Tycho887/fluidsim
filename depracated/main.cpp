#include <iostream>
#include <vector>
#include <random>
#include <cstdint>  // for uint8_t
#include <chrono>   // for timing

// Function to generate a matrix, return pointer to the matrix
uint8_t** generateMatrix(int rows, int cols) {
    // Allocate memory for rows
    uint8_t** matrix = new uint8_t*[rows];
    
    // Allocate memory for each row
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new uint8_t[cols];
    }

    // Initialize random number generator for values between 0 and 100
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 100);

    // Fill the matrix with random numbers
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<uint8_t>(dis(gen));
        }
    }

    return matrix;
}

// Function to clean up the dynamically allocated matrix
void deleteMatrix(uint8_t** matrix, int rows) {
    // Delete each row
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }

    // Delete the array of row pointers
    delete[] matrix;
}

// Function to sum all the elements of the matrix
float sum(uint8_t** matrix, int rows, int cols) {
    float total = 0.0f;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            total += matrix[i][j];
        }
    }
    return total;
}

// Function to calculate the mean of all elements in the matrix
float mean(uint8_t** matrix, int rows, int cols) {
    float total = sum(matrix, rows, cols);
    return total / (rows * cols);  // Total number of elements in the matrix
}

// Function to calculate the variance of all elements in the matrix
float variance(uint8_t** matrix, int rows, int cols) {
    float m = mean(matrix, rows, cols);
    float var = 0.0f;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = matrix[i][j] - m;
            var += diff * diff;
        }
    }
    return var / (rows * cols);
}

int main() {
    const int rows = 5000;
    const int cols = 5000;

    // Start time for matrix generation
    auto start = std::chrono::high_resolution_clock::now();
    
    // Generate the matrix
    uint8_t** matrix = generateMatrix(rows, cols);
    
    // End time for matrix generation
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Matrix generation took: " << duration.count() << " seconds." << std::endl;

    // Start time for mean calculation
    start = std::chrono::high_resolution_clock::now();
    
    // Calculate the mean of the matrix
    float m = mean(matrix, rows, cols);
    
    // End time for mean calculation
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Mean calculation took: " << duration.count() << " seconds." << std::endl;

    // Start time for variance calculation
    start = std::chrono::high_resolution_clock::now();
    
    // Calculate the variance of the matrix
    float v = variance(matrix, rows, cols);
    
    // End time for variance calculation
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Variance calculation took: " << duration.count() << " seconds." << std::endl;

    // Output the mean and variance
    std::cout << "Mean of the matrix: " << m << std::endl;
    std::cout << "Variance of the matrix: " << v << std::endl;

    // Example: Print the first 10 values of the first row
    std::cout << "First 10 values in the first row: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(matrix[0][i]) << " ";
    }
    std::cout << std::endl;

    // Clean up the dynamically allocated memory
    deleteMatrix(matrix, rows);

    return 0;
}
