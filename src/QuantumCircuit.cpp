
#include "../include/QuantumCircuit.h"

#include <iostream>
#include <random>
#include <ctype.h>
#include "../include/LinearAlgebra.h"

using std::cout;
using std::endl;

vector<string> Q1_NAMES = { "H", "X", "Y", "Z", "I", "PHASE", "T", "SQRT-NOT" };// , "R"}


QuantumCircuit::QuantumCircuit(unsigned int num_qubits)
{
	m_num_qubits = num_qubits;
	m_circuit_levels = vector<vector<GateLevel>>(10);
	m_circuit_operator.set_dims(POW_2(m_num_qubits), POW_2(m_num_qubits));
	m_circuit_operator.createIdentityMatrix();
}

void QuantumCircuit::add_gate(unsigned int level, string gate_type, initializer_list<unsigned int> qubits)
{
	m_circuit_levels.at(level).push_back(GateLevel{ level, gate_type, (vector<unsigned int>)qubits });
	//m_circuit_levels.push_back(GateLevel{level, gate_type, (vector<unsigned int>)qubits});
	m_functional_size = level + 1;
}

void QuantumCircuit::populate_1Q(int seed)
{
	std::default_random_engine RAND_NUM_GENERATOR{ static_cast<long unsigned int>(seed) };
	std::uniform_int_distribution<int> distribution(0, Q1_NAMES.size() - 1);

	for (unsigned int i = 0; i < 10; i++)
	{
		for (unsigned int j = 0; j < m_num_qubits; j++)
		{
			int gate_index = distribution(RAND_NUM_GENERATOR);
			string gate_name = Q1_NAMES.at(gate_index);

			this->add_gate(i, gate_name, { j });
		}
	}
}

void QuantumCircuit::print()
{
	cout << "GATES IN THE CIRCUIT" << endl;

	for (unsigned int i = 0; i < m_functional_size; i++)
	{
		if (!m_circuit_levels.at(i).empty())
		{
			cout << "CIRCUIT LEVEL: " << i << endl;

			for (unsigned int j = 0; j < m_circuit_levels.at(i).size(); j++)
			{
				cout << "GATE: " << m_circuit_levels.at(i).at(j).gate_type << ", QUBITS: ";

				for (unsigned int k = 0; k < m_circuit_levels.at(i).at(j).qubit_list.size(); k++)
				{
					cout << m_circuit_levels.at(i).at(j).qubit_list.at(k) << " ";
				}
				cout << endl;
			}
		}
	}
}

void QuantumCircuit::display()
{
	vector<string> circuit_display;
	vector<string> circuit_icons;

	for (unsigned int i = 0; i < m_num_qubits; i++)
	{
		circuit_icons.push_back("-*----*----*----*----*----*----*----*----*----*-");
	}

	for (unsigned int level = 0; level < m_functional_size; level++)
	{
		for (unsigned int gate_index = 0; gate_index < m_circuit_levels.at(level).size(); gate_index++)
		{
			string gate_type = m_circuit_levels.at(level).at(gate_index).gate_type;

			// Single qubit gate
			if (m_circuit_levels.at(level).at(gate_index).qubit_list.size() == 1)
			{
				unsigned int applied_qubit = m_circuit_levels.at(level).at(gate_index).qubit_list.at(0);

				if (gate_type == "H")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'H';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "X")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'X';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "Y")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'Y';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "Z")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'Z';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "I")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'I';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "PHASE")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'P';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "T")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'T';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "SQRT-NOT")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = '!';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
				else if (gate_type == "R")
				{
					circuit_icons.at(applied_qubit).at(5 * level + 1) = 'R';
					circuit_icons.at(applied_qubit).at(5 * level) = '[';
					circuit_icons.at(applied_qubit).at(5 * level + 2) = ']';
				}
			}

			// Two-qubit gate
			else if (m_circuit_levels.at(level).at(gate_index).qubit_list.size() == 2)
			{
				unsigned int control_qubit = m_circuit_levels.at(level).at(gate_index).qubit_list.at(0);
				unsigned int target_qubit = m_circuit_levels.at(level).at(gate_index).qubit_list.at(1);

				if (gate_type == "CNOT")
				{
					circuit_icons.at(control_qubit).at(5 * level + 1) = '+';
					circuit_icons.at(target_qubit).at(5 * level + 1) = '@';
				}
				else if (gate_type == "SWAP")
				{
					circuit_icons.at(control_qubit).at(5 * level + 1) = 'x';
					circuit_icons.at(target_qubit).at(5 * level + 1) = 'x';
				}
				else if (gate_type == "SQRT-SWAP")
				{
					circuit_icons.at(control_qubit).at(5 * level + 1) = 'x';
					circuit_icons.at(target_qubit).at(5 * level + 1) = 'x';
				}
				else if (gate_type == "C-U")
				{
					circuit_icons.at(control_qubit).at(5 * level + 1) = '+';
					circuit_icons.at(target_qubit).at(5 * level + 1) = 'U';
				}
			}

			// Three-qubit gate
			else if (m_circuit_levels.at(level).at(gate_index).qubit_list.size() == 3)
			{
				cout << "Three qubit gate..." << endl;
			}

			// N-qubit gate
			else if (m_circuit_levels.at(level).at(gate_index).qubit_list.size() > 3)
			{
				cout << "N qubit gate..." << endl;
			}

			// Zero-qubit gate - ERROR STATE
			else
			{
				cout << "Error here" << endl;
			}
		}
	}

	// Replace the debugging asterisks with dashes
	for (unsigned int i = 0; i < circuit_icons.size(); i++)
	{
		for (unsigned int j = 0; j < circuit_icons.at(i).size(); j++)
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
	for (unsigned int i = 0; i < m_num_qubits * 2; i++)
	{
		// If the line is a qubit line, print the contents of the icon vector
		if (i % 2 == 0)
		{
			string qubit_str = "Q[" + std::to_string(i / 2) + "] -----" + circuit_icons.at(i / 2) + "-----";
			circuit_display.push_back(qubit_str);
		}

		// Else the line is a mid line, print mostly empty space except for multi-qubit gates
		else if (i % 2 == 1)
		{
			unsigned int qubit = i / 2;
			string mid_line;

			for (unsigned int j = 0; j < circuit_icons.at(qubit).size(); j++)
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

	for (unsigned int i = 0; i < circuit_display.size(); i++)
	{
		cout << circuit_display[i] << endl;
	}
}

void QuantumCircuit::initialize(initializer_list<unsigned int> qubits, State& init_state)
{
	vector<unsigned int> qubit_list = (vector<unsigned int>)qubits;

	for (unsigned int i = 0; i < m_num_qubits; i++)
	{
		
	}
}

// Sort the qubits in each level according to ascending order
// Basically need at least one of the qubits of gate0 to be smaller than the qubits of gate1...gateN

void QuantumCircuit::sort_levels()
{
	unsigned int min_qubit = 0;
	unsigned int check_qubit = 0;
	unsigned int min_qubit_index = 0;
	GateLevel minGate;
	GateLevel TEMP;
	bool swap_flag = false;

	// For each level in the circuit
	for (unsigned int i = 0; i < m_functional_size; i++)
	{
		// Only perform sorting if more than 1 gate in the level
		if (m_circuit_levels.at(i).size() > 1)
		{
			// For each gate in the level
			for (unsigned int j = 0; j < m_circuit_levels.at(i).size(); j++)
			{
				GateLevel currentGate = m_circuit_levels.at(i).at(j);
				minGate = currentGate;

				// Get the minimum indexed qubit
				if (currentGate.qubit_list.size() == 1)
				{
					min_qubit = currentGate.qubit_list.at(0);
				}
				else if (currentGate.qubit_list.size() == 2)
				{
					min_qubit = (currentGate.qubit_list.at(0) < currentGate.qubit_list.at(1)) ? currentGate.qubit_list.at(0) : currentGate.qubit_list.at(1);
				}

				// For each remaining gate in the level
				for (unsigned int k = j; k < m_circuit_levels.at(i).size(); k++)
				{
					GateLevel checkGate = m_circuit_levels.at(i).at(k);

					// Get the minimum indexed qubit
					if (checkGate.qubit_list.size() == 1)
					{
						check_qubit = checkGate.qubit_list.at(0);
					}
					else if (checkGate.qubit_list.size() == 2)
					{
						check_qubit = (checkGate.qubit_list.at(0) < checkGate.qubit_list.at(1)) ? checkGate.qubit_list.at(0) : checkGate.qubit_list.at(1);
					}

					// If new minimum qubit found, update minimum information
					if (check_qubit < min_qubit)
					{
						min_qubit = check_qubit;
						min_qubit_index = k;
						minGate = checkGate;
						swap_flag = true;
					}
				}
				// Perform the swap for ordering, if required
				if (swap_flag)
				{
					TEMP = m_circuit_levels.at(i).at(j);
					m_circuit_levels.at(i).at(j) = minGate;
					m_circuit_levels.at(i).at(min_qubit_index) = TEMP;
					swap_flag = false;
				}
			}
		}
	}
}

string TENSOR_SUBSTR = "-I";

void QuantumCircuit::evolve()
{
	sort_levels();

	vector<vector<string>> circuit_compile_strings;
	vector<vector<Operator>> level_ops;
	vector<Operator> level_matrix;
	level_matrix.reserve(m_functional_size);

	vector<string> level_compile_strings;
	string base_string;
	unsigned int NUM_COMPILE_STRS = 1;

	// For each level in the circuit
	for (unsigned int i = 0; i < m_functional_size; i++)
	{
		// Reset the strings
		NUM_COMPILE_STRS = 1;
		level_compile_strings.clear();
		base_string.clear();
		base_string.reserve(TENSOR_SUBSTR.length() * m_num_qubits);

		for (unsigned int k = 0; k < m_num_qubits; k++)
		{
			base_string.append(TENSOR_SUBSTR);
		}
		// FIRST PASS THROUGH THE LEVEL - DETERMINE THE NUMBER OF COMPILE STRINGS NEEDED, AND BUILD THE COMPILE STRING WITH ALL 1Q GATES IN THE LEVEL

		// For each gate in the level
		for (unsigned int j = 0; j < m_circuit_levels.at(i).size(); j++)
		{
			GateLevel currentGate = m_circuit_levels.at(i).at(j);

			// Populate the base string with all 1Q gates
			if (currentGate.qubit_list.size() == 1)
			{
				base_string.at(2 * currentGate.qubit_list.at(0) + 1) = currentGate.gate_type.at(0);
			}
			// Determine the number of compile strings needed based on the number of control and swap style gates used in the level
			// Need 2 strings for every control gate
			// Need 4 strings for every swap gate
			else if (currentGate.qubit_list.size() == 2)
			{
				if (currentGate.gate_type == "CNOT" || currentGate.gate_type == "C-U")
				{
					NUM_COMPILE_STRS *= 2;
				}
				else if (currentGate.gate_type == "SWAP" || currentGate.gate_type == "SQRT-SWAP")
				{
					NUM_COMPILE_STRS *= 4;
				}
				else
				{
					cout << "UNKNOWN 2Q GATE WEHRSRHSRHJRHT" << endl;
				}
			}
			
		}

		// SECOND PASS: BUILD THE COMPILE STRING VECTOR, SECOND PASS FOR 2Q GATES
		level_compile_strings = vector<string>(NUM_COMPILE_STRS);

		for (unsigned int k = 0; k < level_compile_strings.size(); k++)
		{
			level_compile_strings.at(k) = base_string;
		}
		unsigned int multiplicity = NUM_COMPILE_STRS;
		// For each gate in the level
		for (unsigned int j = 0; j < m_circuit_levels.at(i).size(); j++)
		{
			GateLevel currentGate = m_circuit_levels.at(i).at(j);
			// KNOWN: Control Gate = |0><0| * I + |1><1| * U --> (0 * I) + (3 * X)
			// KNOWN: SWAP Gate = |0><0| * |0><0| + |1><1| * |1><1| + |1><0| * |0><1| + |0><1| * |1><0| --> (0 * 0) + (3 * 3) + (2 * 1) + (1 * 2)
			// Treat like Boolean Algebra problem - Essentially building a  symbolic table
			// Need to care about multiplicity - Basically, fill MSQ first, SHOULD VARY THE LEAST

			// :)
			if (currentGate.qubit_list.size() == 2)
			{
				unsigned int qubit0 = currentGate.qubit_list.at(0);
				unsigned int qubit1 = currentGate.qubit_list.at(1);

				if (currentGate.gate_type == "CNOT" || currentGate.gate_type == "C-U")
				{
					multiplicity /= 2;

					unsigned int periodicity = multiplicity;

					if (multiplicity > 1)
					{
						unsigned int count = 0;
						for (unsigned int k = 0; k < level_compile_strings.size(); k++)
						{
							if (count % 2 == 0)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '0';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = 'I';
							}

							else if (count % 2 == 1)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '3';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = 'U';
							}

							if (k % periodicity == (periodicity - 1))
							{
								count++;
							}

						}
					}
					else if (multiplicity == 1)
					{
						for (unsigned int k = 0; k < level_compile_strings.size(); k++)
						{
							if (k % 2 == 0)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '0';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = 'I';
							}
							else if (k % 2 == 1)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '3';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = 'U';
							}
						}
					}
				}
				else if (currentGate.gate_type == "SWAP" || currentGate.gate_type == "SQRT-SWAP")
				{
					multiplicity /= 4;

					unsigned int periodicity = multiplicity;

					if (multiplicity > 1)
					{
						unsigned int count = 0;

						for (unsigned int k = 0; k < level_compile_strings.size(); k++)
						{
							if (count % 4 == 0)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '0';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '0';
							}
							else if (count % 4 == 1)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '3';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '3';
							}
							else if (count % 4 == 2)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '2';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '1';
							}
							else if (count % 4 == 3)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '1';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '2';
							}

							if (k % periodicity == (periodicity - 1))
							{
								count++;
							}
						}
					}
					else if (multiplicity == 1)
					{
						for (unsigned int k = 0; k < level_compile_strings.size(); k++)
						{
							if (k % 4 == 0)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '0';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '0';
							}
							else if (k % 4 == 1)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '3';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '3';
							}
							else if (k % 4 == 2)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '2';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '1';
							}
							else if (k % 4 == 3)
							{
								level_compile_strings.at(k).at(2 * qubit0 + 1) = '1';
								level_compile_strings.at(k).at(2 * qubit1 + 1) = '2';
							}
						}
					}
				}
				else
				{
					cout << "UNKNOWN 2Q GATE WEHRSRHSRHJRHT" << endl;
				}
			}
		}

		//for (unsigned int k = 0; k < level_compile_strings.size(); k++)
		//{
			//level_compile_strings.at(k) = level_compile_strings.at(k).substr(1);
			//cout << level_compile_strings.at(k) << endl;
		//}
		circuit_compile_strings.push_back(level_compile_strings);
	}

	// For each level in the circuit
	for (unsigned int i = 0; i < circuit_compile_strings.size(); i++)
	{
		vector<Operator> current_level_ops;
		current_level_ops.reserve(circuit_compile_strings.at(i).size());

		//cout << "---------- LEVEL " << i << " ----------" << endl;
		
		// For each compile string in the level
		for (unsigned int j = 0; j < circuit_compile_strings.at(i).size(); j++)
		{
			string current_compile_string = circuit_compile_strings.at(i).at(j);
			//cout << current_compile_string << endl;

			Operator LEVEL_OP(2, 2);
			Operator NEW_LEVEL_OP(2, 2);
			bool first_gate = true;
			// For each gate in the compile string
			for (unsigned int k = 0; k < current_compile_string.length(); k++)
			{
				char current_char = current_compile_string.at(k);
				Gate1Q GATE_OP;
				bool is_gate_char = true;
				unsigned int qubit;
				string lookup_type;
				
				//cout << "----------------- CURRENT CHAR: " << current_char << endl;
				switch (current_char)
				{
					case '-': 
						is_gate_char = false;
						break;
					case 'X':
						GATE_OP.set_gate("X");
						break;
					case 'Y':
						GATE_OP.set_gate("Y");
						break;
					case 'Z':
						GATE_OP.set_gate("Z");
						break;
					case 'H':
						GATE_OP.set_gate("H");
						break;
					case 'P':
						GATE_OP.set_gate("PHASE");
						break;
					case 'T':
						GATE_OP.set_gate("T");
						break;
					case 'S':
						GATE_OP.set_gate("SQRT-NOT");
						break;
					// If U gate, then need to find the corresponding U operation (i.e. controlled X, controlled Y, etc - found in the gate type!
					case 'U':

						lookup_type;
						qubit = (k - 1) / 2;

						// Look through each gate in the level until the target qubit is found
						for (unsigned int m = 0; m < m_circuit_levels.at(i).size(); m++)
						{
							if (m_circuit_levels.at(i).at(m).qubit_list.at(0) == qubit || m_circuit_levels.at(i).at(m).qubit_list.at(1) == qubit)
							{
								lookup_type = m_circuit_levels.at(i).at(m).gate_type;
							}
						}

						if (lookup_type == "CNOT")
						{
							lookup_type = "X";
						}

						GATE_OP.set_gate(lookup_type);

						break;
					case '0':
						GATE_OP.set_gate("0");
						break;
					case '1':
						GATE_OP.set_gate("1");
						break;
					case '2':
						GATE_OP.set_gate("2");
						break;
					case '3':
						GATE_OP.set_gate("3");
						break;
					case 'I': default:
						GATE_OP.set_gate("I");
						break;	
				}

				if (is_gate_char)
				{
					if (first_gate)
					{
						LEVEL_OP = GATE_OP;
						first_gate = false;
					}
					else
					{
						cout << "LEVEL: " << i << ", GATE: " << k << endl;
						tensor_product(NEW_LEVEL_OP, LEVEL_OP, GATE_OP);
						LEVEL_OP = NEW_LEVEL_OP;
						//LEVEL_OP.print();
					}
				}
			}
			current_level_ops.push_back(LEVEL_OP);
		}
		level_ops.push_back(current_level_ops);
	}

	for (unsigned int m = 0; m < level_ops.size(); m++)
	{
		Operator TEMP(m_num_qubits, m_num_qubits);
		for (unsigned int n = 0; n < level_ops.at(m).size(); n++)
		{
			if (n == 0)
			{
				TEMP = level_ops.at(m).at(0);
			}
			else
			{
				TEMP += level_ops.at(m).at(n);
			}
		}
		level_matrix.push_back(TEMP);
	}

	for (unsigned int i = 0; i < level_matrix.size(); i++)
	{
		cout << "LEVEL: " << i << endl;
		//level_matrix.at(i).print();

		m_circuit_operator *= level_matrix.at(i);

	}

	//m_circuit_operator.print();
	

	

}