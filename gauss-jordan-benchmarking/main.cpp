//
//  main.cpp
//  gauss-jordan
//
//  Created by mndx on 17/04/2022.
//

#include <chrono>

#include "lib_gauss.hpp"
#include "lib_mat.hpp"
#include "lib_mem.hpp"
#include "lib_testing.hpp"
#include "lib_testing_ref.hpp"
#include "user_types.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, char * argv[]) {

    // Size input matrix
    int n = 5;

    // Allocate space for matrices
    double ** mat = mat2D(n);
    double ** mat_inv = mat2D(n);
    double ** mat_inv_ref = mat2D(n);
    double ** mat_prod = mat2D(n);
    double ** mat_store = mat2D(n);
    matrix mat1(n, n);
    i_real_matrix mat2;

    // Populate matrix mat with some data
    init_mat(n, mat);

    // Populate reference matrix mat1 with mat data
    set_mat_to_matrix(mat, n, mat1);

    // Populate reference matrix mat2 with mat data
    init_vec2D(mat, n, mat2);

    // Store initial matrix mat
    set_mat(mat, n, mat_store);

    // Time custom Gauss-Jordan method
    auto start = high_resolution_clock::now();

    // Compute inverse using custom Gauss-Jordan method
    gauss_jordan(mat, n, mat_inv);

    // Get stop time custom Gauss-Jordan method
    auto stop = high_resolution_clock::now();

    // Get duration custom Gauss-Jordan method
    auto duration = duration_cast<seconds>(stop - start);

    // Print duration custom Gauss-Jordan method
    cout << "duration custom Guass-Jordan: " << duration.count() << " (s)" << endl;

    // Time reference method 1
    start = high_resolution_clock::now();

    // Compute inverse using reference method 1
    auto mat1_inv = inverse(mat1);

    // Get stop time reference method 1
    stop = high_resolution_clock::now();

    // Get duration reference method 1
    duration = duration_cast<seconds>(stop - start);

    // Print duration reference method 1
    cout << "duration reference method 1: " << duration.count() << " (s)" << endl;

    // Time reference method 2
    start = high_resolution_clock::now();

    // Compute inverse using reference method 2
    i_real_matrix mat2_inv = inv_ref(mat2, true);

    // Get stop time reference method 2
    stop = high_resolution_clock::now();

    // Get duration reference method 2
    duration = duration_cast<seconds>(stop - start);

    // Print duration of reference method 2
    cout << "duration reference method 2: " << duration.count() << " (s)" << endl;

    // Verify computation custom Gauss-Jordan method
    mat_mult_sq(mat_store, mat_inv, n, mat_prod);

    // Print results
    print_mat(mat_prod, n);

    print_mat(mat_inv, n);

    print(std::cout, mat1_inv);

    showMatrix(mat2_inv, "reference solution", false);

    // Free allocated space
    free_mat2D(mat, n);
    free_mat2D(mat_inv, n);
    free_mat2D(mat_inv_ref, n);
    free_mat2D(mat_prod, n);
    free_mat2D(mat_store, n);

    return 0;
}



