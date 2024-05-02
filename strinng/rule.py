#    Copyright 2024 Alexander Koziell-Pipe

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Rewrite Rule Class."""
from __future__ import annotations
from copy import deepcopy

from strinng.hypergraph import Hypergraph


class RuleError(Exception):
    """An error occurred in the rule logic."""
    pass


class Rule:
    """A hypergraph rewrite rule."""

    def __init__(self, lhs: Hypergraph, rhs: Hypergraph,
                 name: str, is_equiv: bool) -> None:
        if lhs.domain() != rhs.domain():
            raise RuleError('Domain must match on lhs and rhs of'
                            + ' rule.')
        if lhs.codomain() != rhs.codomain():
            raise RuleError('Domain must match on lhs and rhs of'
                            + ' rule.')
        self.lhs = lhs
        self.rhs = rhs
        self.name = name
        self.is_equiv = is_equiv

    def converse(self) -> Rule:
        """Return the converse of this rule"""
        if self.name.startswith('-'):
            self.name = self.name[1:]
        else:
            name = '-' + self.name

        return Rule(deepcopy(self.rhs), deepcopy(self.lhs),
                    name, True)

    def is_left_linear(self) -> bool:
        """Return whether this rule is left linear.

        This means the boundary on the lhs embeds injectively.
        """
        vertices = set()
        for vertex in self.lhs.inputs + self.rhs.outputs:
            if vertex in vertices:
                return False
            vertices.add(vertex)
        return True
