
#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <complex>
#include <math.h>
#include <string>

struct mat_coord_t
{
	unsigned int row;
	unsigned int col;
};

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

unsigned int RC_TO_INDEX(unsigned int row, unsigned int col, unsigned int stride);

unsigned int POW_2(unsigned int x);

const double PI = std::atan(1.0) * 4;
const double INV_SQRT2 = (1.0 / std::sqrt(2));

std::string complexDoubleToString(std::complex<double>& z);

std::complex<double> stringToComplexDouble(std::string z_str);

#endif //UTILITY_H
