
#ifndef GATE_H
#define GATE_H

#include "Operator.h"

class Gate : public Operator
{
public:
	Gate();
	~Gate();

	void print();
protected:
	string m_name;
};


class Gate1Q : public Gate
{

public:

	Gate1Q();
	Gate1Q(const string gate_type);
	Gate1Q(const string gate_type, complex<double> phase_arg);

	void set_gate(const string gate_type);

	~Gate1Q();
};

class Gate2Q : public Gate
{
public:

	Gate2Q(const string gate_type);

	~Gate2Q();
};

#endif // GATE_H

