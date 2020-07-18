
#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <complex>

// Is Zero for printing purposes (i.e. maybe just less than 1e-5, just to het general shape and fit on screen)
bool iszero_print(const std::complex<double>& z);

// Format and print complex numbers of the form z = a + bi

// If z is pure real --> print a
// If z is pure imag --> print bi
// If z is complex --> print a + bi
std::ostream& operator<<(std::ostream& os, const std::complex<double>& z);

// Complex equality to 0
// Needs to fall within zero radius (< 1e-20)?
bool iszero(const std::complex<double>& z);

std::ostream& operator<<(std::ostream& os, const mat_coord_t& coord_pair);

std::complex<double> sign(const std::complex<double>& z);

const double PI = std::atan(1.0) * 4;
const double INV_SQRT2 = (1.0 / std::sqrt(2));

struct mat_coord_t
{
	unsigned int row;
	unsigned int col;
};

#endif //UTILITY_H