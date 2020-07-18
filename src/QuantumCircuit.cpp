
#include "../include/QuantumCircuit.h"

#include <iostream>
#include <ctype.h>

using std::cout;
using std::endl;

QuantumCircuit::QuantumCircuit(unsigned int num_qubits)
{
	m_num_qubits = num_qubits;
}

void QuantumCircuit::add_gate(unsigned int level, string gate_type, initializer_list<unsigned int> qubits)
{
	m_circuit_levels.push_back(GateLevel{level, gate_type, (vector<unsigned int>)qubits});
}

void QuantumCircuit::print()
{
	cout << "GATES IN THE CIRCUIT" << endl;

	for (int i = 0; i < m_circuit_levels.size(); i++)
	{
		cout << "CIRCUIT LEVEL: " << m_circuit_levels[i].level << ", GATE: " << m_circuit_levels[i].gate_type << ", QUBITS: ";

		for (int j = 0; j < m_circuit_levels[i].qubit_list.size(); j++)
		{
			 cout << m_circuit_levels[i].qubit_list[j] << " ";
		}
		cout << endl;
	}
}

void QuantumCircuit::display()
{
	vector<string> circuit_display;
	vector<string> circuit_icons;

	for (int i = 0; i < m_num_qubits; i++)
	{
		circuit_icons.push_back("-*----*----*----*----*----*----*----*----*----*");
	}

	for (int entry = 0; entry < m_circuit_levels.size(); entry++)
	{
		string gate_type = m_circuit_levels.at(entry).gate_type;

		unsigned int level = m_circuit_levels.at(entry).level;

		// Single qubit gate
		if (m_circuit_levels.at(entry).qubit_list.size() == 1)
		{
			unsigned int applied_qubit = m_circuit_levels.at(entry).qubit_list.at(0);

			if (gate_type == "HADAMARD")
			{
				circuit_icons.at(applied_qubit).at(5 * level + 1) = 'H';
				circuit_icons.at(applied_qubit).at(5 * level) = '[';
				circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
			}
		}

		// Two-qubit gate
		else if (m_circuit_levels.at(entry).qubit_list.size() == 2)
		{
			unsigned int control_qubit = m_circuit_levels.at(entry).qubit_list.at(0);
			unsigned int target_qubit = m_circuit_levels.at(entry).qubit_list.at(1);

			if (gate_type == "CNOT")
			{
				circuit_icons.at(control_qubit).at(5 * level + 1) = '+';
				circuit_icons.at(target_qubit).at(5 * level + 1) = '@';
			}
		}


		// Three-qubit gate
		else if (m_circuit_levels.at(level).qubit_list.size() == 3)
		{
			cout << "Three qubit gate..." << endl;
		}

		// N-qubit gate
		else if (m_circuit_levels.at(level).qubit_list.size() > 3)
		{
			cout << "N qubit gate..." << endl;
		}

		// Zero-qubit gate - ERROR STATE
		else
		{
			cout << "Error here" << endl;
		}
	}

	// Replace the debugging asterisks with dashes
	for (int i = 0; i < circuit_icons.size(); i++)
	{
		for (int j = 0; j < circuit_icons.at(i).size(); j++)
		{
			if (circuit_icons.at(i).at(j) == '*')
			{
				circuit_icons.at(i).at(j) = '-';
			}
		}
	}


	//for (int i = 0; i < circuit_icons.size(); i++)
	//{
		//cout << circuit_icons[i] << endl;
	//}

	//system("PAUSE");

	cout << endl;
	for (int i = 0; i < m_num_qubits * 2; i++)
	{
		// If the line is a qubit line, print the contents of the icon vector
		if (i % 2 == 0)
		{
			string qubit_str = "Q[" + std::to_string(i / 2) + "] -----" + circuit_icons.at(i / 2);
			circuit_display.push_back(qubit_str);
		}

		// Else the line is a mid line, print mostly empty space except for multi-qubit gates
		else if (i % 2 == 1)
		{
			unsigned int qubit = i / 2;
			string mid_line;

			for (int j = 0; j < circuit_icons.at(qubit).size(); j++)
			{
				char current_char = circuit_icons.at(qubit).at(j);

				// Insert appropriate character based on content of the icon line
				switch (current_char)
				{
					// Empty
					case '[': case ']': mid_line.push_back(' '); break;
					case '*': case '-': mid_line.push_back(' '); break;
					// Multi-level qubit gate encountered
					case '+': mid_line.push_back('|'); break;
					default: mid_line.push_back(' ');
				}

				// If character is represented by a letter, it is most likely a gate - just push back
				if (isalpha(current_char))
				{
					//mid_line.push_back('&');
				}
			}
			circuit_display.push_back("          " + mid_line);
		}
	}

	for (int i = 0; i < circuit_display.size(); i++)
	{
		cout << circuit_display[i] << endl;
	}
}

void QuantumCircuit::initialize(initializer_list<unsigned int> qubits, State& init_state)
{
	vector<unsigned int> qubit_list = (vector<unsigned int>)qubits;

	for (int i = 0; i < m_num_qubits; i++)
	{
		
	}
}