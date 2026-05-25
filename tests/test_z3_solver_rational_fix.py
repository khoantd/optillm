#!/usr/bin/env python3
"""Tests for Z3/SymPy solver symbolic Rational fix."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.z3_solver import fix_symbolic_rational_calls, execute_code_in_process


class TestSymbolicRationalFix(unittest.TestCase):
    def test_rewrites_symbolic_rational_to_division(self):
        code = "from sympy import symbols, Rational\np, q = symbols('p q')\nx = Rational(p, q)\n"
        fixed = fix_symbolic_rational_calls(code)
        self.assertIn("p / q", fixed)
        self.assertNotIn("Rational(p, q)", fixed)

    def test_keeps_numeric_rational(self):
        code = "from sympy import Rational\nx = Rational(2, 3)\n"
        fixed = fix_symbolic_rational_calls(code)
        self.assertIn("Rational(2, 3)", fixed)

    def test_execute_sqrt2_irrationality_snippet(self):
        code = """from sympy import symbols, Eq, sqrt, Rational, simplify
p, q = symbols('p q', integer=True)
assumption = Eq(sqrt(2), Rational(p, q))
equation = Eq(p**2, 2 * q**2)
k = symbols('k', integer=True)
substituted_equation = equation.subs(p, 2*k)
simplified_equation = simplify(substituted_equation)
print(simplified_equation)
"""
        status, output = execute_code_in_process(fix_symbolic_rational_calls(code))
        self.assertEqual(status, "success")
        self.assertIn("4*k**2", output.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
