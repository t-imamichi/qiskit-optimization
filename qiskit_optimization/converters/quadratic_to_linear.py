# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A converter from quadratic terms to linear terms."""

import copy
from typing import cast

import numpy as np

from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_expression import QuadraticExpression
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable, VarType
from .quadratic_program_converter import QuadraticProgramConverter


class QuadraticToLinear(QuadraticProgramConverter):
    """Convert quadratic terms of binary variables into linear terms."""

    _PREFIX = "_and"

    def __init__(self) -> None:
        self._src: QuadraticProgram | None = None
        self._dst: QuadraticProgram | None = None
        self._and: dict[tuple[str, str], str] = {}

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with quadratic terms of binary variables into linear terms.

        More specifically, a quadratic term ``x * y`` of binary variables ``x`` and ``y``
        will be replaced with a new binary variable ``_x_y`` and the following linear constraints
        ensures ``_x_y == x * y`` (aka. McCormick linearization).

        .. parsed-literal:::
            _x_y <= x
            _x_y <= y
            _x_y >= x + y - 1
            x, y, _x_y: binary variables

        Args:
            problem: The problem to be solved, that may contain inequality constraints.

        Returns:
            The converted problem, that contain only equality constraints.

        Raises:
            QiskitOptimizationError: If a variable type is not supported.
        """
        self._src = copy.deepcopy(problem)
        self._dst = QuadraticProgram(name=problem.name)
        self._and.clear()

        # copy variables
        for x in self._src.variables:
            if x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound)
            elif x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(
                    name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound
                )
            else:
                raise QiskitOptimizationError(f"Unsupported variable type {x.vartype}")

        # add binary variables representing quadratic terms and additional constraints
        self._linearize_add_vars(self._src.objective.quadratic)
        for quad_const in self._src.quadratic_constraints:
            self._linearize_add_vars(quad_const.quadratic)
        self._linearize_add_constraints()

        # copy linear constraints
        for lin_const in self._src.linear_constraints:
            self._dst.linear_constraint(
                lin_const.linear.to_dict(), lin_const.sense, lin_const.rhs, lin_const.name
            )

        # convert objective function
        constant = self._src.objective.constant
        linear = self._src.objective.linear.to_dict(use_name=True)
        linear.update(self._linearize_convert(self._src.objective.quadratic))
        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear)
        else:
            self._dst.maximize(constant, linear)

        # convert quadratic constraints
        for quad_const in self._src.quadratic_constraints:
            linear = quad_const.linear.to_dict(use_name=True)
            linear.update(self._linearize_convert(quad_const.quadratic))
            self._dst.linear_constraint(
                linear,
                quad_const.sense,
                quad_const.rhs,
                quad_const.name,
            )

        return self._dst

    def _linearize_add_vars(self, expr: QuadraticExpression):
        problem = self._dst
        coeffs = expr.to_dict(use_name=True)
        for var_x, var_y in coeffs.keys():
            var_x = cast(str, var_x)
            var_y = cast(str, var_y)
            if (
                problem.get_variable(var_x).vartype == VarType.BINARY
                and problem.get_variable(var_y).vartype == VarType.BINARY
            ):
                if (var_x, var_y) in self._and:
                    continue
                name = "_".join([self._PREFIX, var_x, var_y])
                self._and[var_x, var_y] = name
                # problem.binary_var(name)
                problem.continuous_var(lowerbound=0, upperbound=1, name=name)

    def _linearize_add_constraints(self):
        problem = self._dst
        for (var_x, var_y), name in self._and.items():
            problem.linear_constraint({name: 1, var_x: -1}, "<=", 0, f"{name}_1")
            problem.linear_constraint({name: 1, var_y: -1}, "<=", 0, f"{name}_2")
            problem.linear_constraint({name: 1, var_x: -1, var_y: -1}, ">=", -1, f"{name}_3")

    def _linearize_convert(self, expr: QuadraticExpression):
        coeffs = expr.to_dict(use_name=True)
        linear = {}
        for (var_x, var_y), coeff in coeffs.items():
            var_x = cast(str, var_x)
            var_y = cast(str, var_y)
            if (var_x, var_y) in self._and:
                linear[self._and[var_x, var_y]] = coeff
        return linear

    def interpret(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.
        """
        # convert back the optimization result into that of the original problem
        names = [var.name for var in self._src.variables]

        # interpret slack variables
        sol = {name: x[i] for i, name in enumerate(names)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            new_x[i] = sol[var.name]
        return new_x
