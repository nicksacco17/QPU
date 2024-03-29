
#include "../include/Utility.h"

#include <iomanip>
#include <iostream>
#include <math.h>

using std::cout;
using std::endl;

bool iszero_print(const std::complex<double>& z)
{
	if (std::abs(z) <= 1e-10)
	{
		return true;
	}
	else
	{
		return false;
	}
}

std::ostream& operator<<(std::ostream& os, const std::complex<double>& z)
{
	if (iszero_print(std::real(z)) && iszero_print(std::imag(z)))
	{
		os << "0 ";
	}
	// Pure real
	else if (iszero_print(std::imag(z)))
	{
		os << std::real(z) << " ";
	}

	// Pure imaginary
	else if (iszero_print(std::real(z)))
	{
		os << std::imag(z) << "i ";
	}

	// Complex number
	else
	{
		os << std::real(z) << " + " << std::imag(z) << "i ";
	}
	return os;
}

// Floating-point zero evaluation
bool iszero(const std::complex<double>& z)
{
	return (std::fpclassify(std::real(z)) == FP_ZERO && std::fpclassify(std::imag(z)) == FP_ZERO);
}

std::complex<double> sign(const std::complex<double>& z)
{
	if (iszero(z))
	{
		return 1.0;
	}
	else
	{
		return (z / std::abs(z));
	}
}

std::ostream& operator<<(std::ostream& os, const mat_coord_t& coord_pair)
{
	os << "(" << coord_pair.row << ", " << coord_pair.col << ")";

	return os;
}

unsigned int RC_TO_INDEX(unsigned int row, unsigned int col, unsigned int stride)
{
	return ((row * stride) + col);
}

unsigned int POW_2(unsigned int x)
{
	return std::pow(2, x);
}

std::string complexDoubleToString(std::complex<double>& z)
{
	std::string z_str = "(" + std::to_string(std::real(z)) + ", " + std::to_string(std::imag(z)) + ")";

	return z_str;
}

std::complex<double> stringToComplexDouble(std::string z_str)
{
	double z_real = std::stod(z_str.substr(1, z_str.find_first_of(',') - 1));
	double z_imag = std::stod(z_str.substr(z_str.find_first_of(' ') + 1, z_str.find_last_of(')') - z_str.find_first_of(' ') - 1));

	std::complex<double> z(z_real, z_imag);
	return z;
}