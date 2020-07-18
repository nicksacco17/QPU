#pragma once

#include <fstream>
#include <complex>

bool iszero_print(const std::complex<double>& z);

// Format and print complex numbers of the form z = a + bi

// If z is pure real --> print a
// If z is pure imag --> print bi
// If z is complex --> print a + bi
std::ostream& operator<<(std::ostream& os, const std::complex<double>& z);

// Complex equality to 0




bool iszero(const std::complex<double>& z);

struct mat_coord_t
{
	unsigned int row;
	unsigned int col;
};

std::ostream& operator<<(std::ostream& os, const mat_coord_t& coord_pair);

std::complex<double> sign(const std::complex<double>& z);

const double PI = std::atan(1.0) * 4;
const double INV_SQRT2 = (1.0 / std::sqrt(2));