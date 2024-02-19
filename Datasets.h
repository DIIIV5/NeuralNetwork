#include <map>

namespace HandWrittenMathSymbols {
	const int Size = 19;
	const char SYMBOL_LIST[Size] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '×', '÷', '=', '.', 'x', 'y', 'z' };
	const std::map<const char, int> SymbolToInt = {
	{ '0', 0 },
	{ '1', 1},
	{ '2', 2 },
	{ '3', 3 },
	{ '4', 4 },
	{ '5', 5 },
	{ '6', 6 },
	{ '7', 7 },
	{ '8', 8 },
	{ '9', 9 },
	{ '+', 10 },
	{ '-', 11 },
	{ '×', 12 },
	{ '÷', 13 },
	{ '=', 14 },
	{ '.', 15 },
	{ 'x', 16 },
	{ 'y', 17 },
	{ 'z', 18 },
	};

	const std::string Root = "data/Handwritten Math Symbols/dataset/";
	
	const std::map<const char, std::string> Path = {
		{ '0', Root + "0" },
		{ '1', Root + "1"},
		{ '2', Root + "2" },
		{ '3', Root + "3" },
		{ '4', Root + "4" },
		{ '5', Root + "5" },
		{ '6', Root + "6" },
		{ '7', Root + "7" },
		{ '8', Root + "8" },
		{ '9', Root + "9" },
		{ '+', Root + "add" },
		{ '-', Root + "sub" },
		{ '×', Root + "mul" },
		{ '÷', Root + "div" },
		{ '=', Root + "eq" },
		{ '.', Root + "dec" },
		{ 'x', Root + "x" },
		{ 'y', Root + "y" },
		{ 'z', Root + "z" },
	};
}