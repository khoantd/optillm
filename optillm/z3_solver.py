from typing import Dict, Any
from z3 import *
import sympy
import io
import re
import contextlib
import logging
import ast
import math
import multiprocessing
import traceback
import optillm
from optillm import conversation_logger

class TimeoutException(Exception):
    pass

def _is_rational_call(func: ast.AST) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "Rational"
    if isinstance(func, ast.Attribute):
        return func.attr == "Rational"
    return False


def _is_numeric_ast(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_numeric_ast(node.operand)
    return False


class _SymbolicRationalFixer(ast.NodeTransformer):
    """Rewrite ``Rational(p, q)`` with symbols to ``p / q`` (LLM misuse)."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not _is_rational_call(node.func) or len(node.args) < 2:
            return node
        numer, denom = node.args[0], node.args[1]
        if _is_numeric_ast(numer) and _is_numeric_ast(denom):
            return node
        if isinstance(numer, ast.Name) and isinstance(denom, ast.Name):
            return ast.copy_location(
                ast.BinOp(left=numer, op=ast.Div(), right=denom),
                node,
            )
        return node


def fix_symbolic_rational_calls(code: str) -> str:
    """Fix LLM code that passes SymPy symbols to ``Rational``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    fixed = _SymbolicRationalFixer().visit(tree)
    ast.fix_missing_locations(fixed)
    return ast.unparse(fixed)


def prepare_execution_globals():
    """
    Prepare globals dictionary for Z3/SymPy code execution.

    WARNING: This is NOT a security sandbox. The name "execution_globals" reflects
    that this simply provides the execution environment for solver code, not a
    security boundary. The code is executed via exec() with access to z3, sympy,
    and math libraries. Only execute trusted code.
    """
    execution_globals = {
        'print': print,
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'float': float,
            'int': int,
            'len': len,
            'max': max,
            'min': min,
            'round': round,
            'sum': sum,
            'complex': complex,
        }
    }
    
    # Add common math functions
    execution_globals.update({
        'log': math.log,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e,
    })

    # Add complex number support
    execution_globals['I'] = complex(0, 1)
    execution_globals['Complex'] = complex

    return execution_globals

def execute_code_in_process(code: str):
    import z3
    import sympy
    import math
    import itertools
    from fractions import Fraction

    execution_globals = prepare_execution_globals()

    # Add Z3 specific functions
    z3_whitelist = set(dir(z3))
    execution_globals.update({name: getattr(z3, name) for name in z3_whitelist})

    # Add SymPy specific functions
    sympy_whitelist = set(dir(sympy))
    execution_globals.update({name: getattr(sympy, name) for name in sympy_whitelist})

    # Ensure key Z3 and SymPy components are available
    execution_globals.update({
        'z3': z3,
        'sympy': sympy,
        'Solver': z3.Solver,
        'solver': z3.Solver,
        'Optimize': z3.Optimize,
        'sat': z3.sat,
        'unsat': z3.unsat,
        'unknown': z3.unknown,
        'Real': z3.Real,
        'Int': z3.Int,
        'Bool': z3.Bool,
        'And': z3.And,
        'Or': z3.Or,
        'Not': z3.Not,
        'Implies': z3.Implies,
        'If': z3.If,
        'Sum': z3.Sum,
        'ForAll': z3.ForAll,
        'Exists': z3.Exists,
        'model': z3.Model,
        'Symbol': sympy.Symbol,
        'solve': sympy.solve,
        'simplify': sympy.simplify,
        'expand': sympy.expand,
        'factor': sympy.factor,
        'diff': sympy.diff,
        'integrate': sympy.integrate,
        'limit': sympy.limit,
        'series': sympy.series,
    })
    
    # Add custom functions
    def as_numerical(x):
        if z3.is_expr(x):
            if z3.is_int_value(x) or z3.is_rational_value(x):
                return float(x.as_decimal(20))
            elif z3.is_algebraic_value(x):
                return x.approx(20)
        return float(x)

    execution_globals['as_numerical'] = as_numerical

    def Mod(x, y):
        return x % y

    execution_globals['Mod'] = Mod

    def Z3Rational(numerator, denominator=1):
        return z3.Real(str(Fraction(numerator, denominator)))

    execution_globals['Z3Rational'] = Z3Rational

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(code, execution_globals, {})
        except Exception:
            return ("error", traceback.format_exc())
    return ("success", output_buffer.getvalue())

class Z3SymPySolverSystem:
    def __init__(self, system_prompt: str, client, model: str, timeout: int = 30, request_config: dict = None, request_id: str = None):
        self.system_prompt = system_prompt
        self.model = model
        self.client = client
        self.timeout = timeout
        self.solver_completion_tokens = 0
        self.request_id = request_id

        # Extract max_tokens from request_config with default
        self.max_tokens = 4096
        if request_config:
            self.max_tokens = request_config.get('max_tokens', self.max_tokens)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def process_query(self, query: str) -> str:
        try:
            analysis = self.analyze_query(query)
            if "SOLVER_CAN_BE_APPLIED: True" not in analysis:
                return self.standard_llm_inference(query), self.solver_completion_tokens
            
            formulation = self.extract_and_validate_expressions(analysis)
            solver_result = self.solve_with_z3_sympy(formulation)
             
            return self.generate_response(query, analysis, solver_result), self.solver_completion_tokens
        except Exception as e:
            logging.error(f"An error occurred while processing the query with Z3 and SymPy, returning standard llm inference results: {str(e)}")
            return self.standard_llm_inference(query), self.solver_completion_tokens

    def analyze_query(self, query: str) -> str:
        analysis_prompt = f"""Analyze the given query and determine if it can be solved using Z3 or SymPy:

1. Identify variables, constraints, and objectives.
2. Determine the problem type (e.g., SAT, optimization, symbolic manipulation).
3. Decide if Z3, SymPy, or a combination of both is suitable.

If Z3 or SymPy can be applied, provide Python code using the appropriate library (or both) to solve the problem. Make sure you define any additional methods you need for solving the problem.
The code will be executed in an environment with Z3 and SymPy available, so do not include any other libraries or modules.

Important SymPy rules:
- For symbolic fractions use ``p/q``, not ``Rational(p, q)`` (``Rational`` is only for numeric literals like ``Rational(2, 3)``).
- Use ``symbols('p q', integer=True)`` or separate ``symbols('p'); symbols('q')`` for multiple variables.

Query: {query}

Respond with:
SOLVER_CAN_BE_APPLIED: [True/False]

SOLVER_FORMULATION:
```python
# Z3 and/or SymPy code here
```

Analysis:
[Your step-by-step analysis]
"""
        
        provider_request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            "max_tokens": 1024,
            "n": 1,
            "temperature": 0.1
        }
        analysis_response = self.client.chat.completions.create(**provider_request)
        
        # Log provider call
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and self.request_id:
            response_dict = analysis_response.model_dump() if hasattr(analysis_response, 'model_dump') else analysis_response
            optillm.conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        
        self.solver_completion_tokens = analysis_response.usage.completion_tokens
        return analysis_response.choices[0].message.content

    def generate_response(self, query: str, analysis: str, solver_result: Dict[str, Any]) -> str:
        if solver_result.get("status") != "success":
            return self.standard_llm_inference(query)
        
        response_prompt = f"""Provide a clear answer to the query using the analysis and solver result:

Query: {query}

Analysis: {analysis}

Solver Result: {solver_result.get("output")}

Response:
"""
        
        provider_request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": response_prompt}
            ],
            "max_tokens": self.max_tokens,
            "n": 1,
            "temperature": 0.1
        }
        response = self.client.chat.completions.create(**provider_request)
        
        # Log provider call
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        
        self.solver_completion_tokens = response.usage.completion_tokens
        return response.choices[0].message.content

    def standard_llm_inference(self, query: str) -> str:
        provider_request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": self.max_tokens,
            "n": 1,
            "temperature": 0.1
        }
        response = self.client.chat.completions.create(**provider_request)
        
        # Log provider call
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and self.request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
        
        self.solver_completion_tokens = response.usage.completion_tokens
        return response.choices[0].message.content

    def extract_and_validate_expressions(self, analysis: str) -> str:
        formulation = re.search(r"```python\n([\s\S]+?)```", analysis)
        if formulation:
            return formulation.group(1).strip()
        raise ValueError("No valid Z3 or SymPy formulation found in the analysis.")

    def solve_with_z3_sympy(self, formulation: str, max_attempts: int = 3) -> Dict[str, Any]:
        for attempt in range(max_attempts):
            output = self.execute_solver_code(formulation)
            if "Error:" not in output:
                return {"status": "success", "output": output}
        
            error_prompt = f"""Fix the Z3 or SymPy code that resulted in an error. Follow these steps:

    1. Review the original code and the error message carefully.
    2. Analyze the error and identify its root cause.
    3. Think through the necessary changes to fix the error.
    4. Generate a corrected version of the code.

    Original Code:
    {formulation}

    Error Message:
    {output}

    Step-by-Step Analysis:
    [Provide your step-by-step analysis here]

    Corrected Z3 or SymPy Code:
    ```python
    # Corrected code here
    ```
    """
            provider_request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": error_prompt}
                ],
                "max_tokens": 1024,
                "n": 1,
                "temperature": 0.1
            }
            response = self.client.chat.completions.create(**provider_request)
            
            # Log provider call
            if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and self.request_id:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                optillm.conversation_logger.log_provider_call(self.request_id, provider_request, response_dict)
            
            self.solver_completion_tokens = response.usage.completion_tokens
            formulation = self.extract_and_validate_expressions(response.choices[0].message.content)

        return {"status": "failed", "output": "Failed to solve after multiple attempts."}

    def execute_solver_code(self, code: str) -> str:
        logging.info("Executing Z3 and SymPy solver code")
        logging.info(f"Code: {code}")
        
        # Parse the code into an AST
        try:
            _ = ast.parse(code)
        except SyntaxError as e:
            logging.error(f"Syntax error in provided code: {e}")
            return f"Error: Syntax error: {e}"

        code = fix_symbolic_rational_calls(code)

        # Execute the code in a separate process
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(1) as pool:
            async_result = pool.apply_async(execute_code_in_process, (code,))
            try:
                status, result = async_result.get(timeout=self.timeout)
            except multiprocessing.TimeoutError:
                pool.terminate()
                logging.error("Execution timed out")
                return "Error: Execution timed out"

        if status == "error":
            logging.error(f"Execution error: {result}")
            return f"Error: {result}"

        logging.info("Z3 and SymPy solver code executed successfully")
        return result