"""Rewrite application function."""
from copy import deepcopy
from typing import Iterable

from strinng.match import Match
from strinng.rule import Rule


def dpo(rule: Rule, match: Match) -> Iterable[Match]:
    """Do a double pushout rewrite.

    `match` must be a match of of `rule.rhs` into a hypergraph.
    """

    in_map: dict[int, int] = dict()
    out_map: dict[int, int] = dict()

    # Create the context
    # This is a hypergraph with a 'hole' where the rewrite occurs.
    context = deepcopy(match.codomain)
    for edge_id in rule.lhs.edges.keys():
        context.remove_edge(edge_id)
    for vertex_id in rule.rhs.vertices.keys():
        new_vertex = match.vertex_map[vertex_id]
        if rule.lhs.is_boundary(new_vertex):
            in_count = sum(vertex_id == i for i in rule.lhs.inputs)
            out_count = sum(vertex_id == o for o in rule.lhs.outputs)
            if in_count == 1 and out_count == 1:
                v1i, v1o = context.explode_vertex(new_vertex)
                if len(v1i) == 1 and len(v1o) == 1:
                    in_map[vertex_id] = v1i[0]
                    out_map[vertex_id] = v1o[0]
                else:
                    raise NotImplementedError(
                        "Rewriting modulo Frobenius not yet supported.")
            elif in_count > 1 or out_count > 1:
                raise NotImplementedError(
                    "Rewriting modulo Frobenius not yet supported.")
        else:
            context.remove_vertex(new_vertex)

    # Match the rhs of the rule into the context
    new_match = Match(rule.rhs, context)

    # Map inputs according to matching of lhs into the context
    for lhs_v, rhs_v in zip(rule.lhs.inputs, rule.rhs.inputs):
        new_match.vertex_map[rhs_v] = (in_map[lhs_v] if lhs_v in in_map
                                       else match.vertex_map[lhs_v])

    # Map the outputs. If the same vertex is an input and an output in
    # rule.rhs, then merge them in the rewritten graph.
    for lhs_v, rhs_v in zip(rule.lhs.outputs, rule.rhs.outputs):
        rhs_v1 = (out_map[lhs_v] if lhs_v in out_map
                  else match.vertex_map[lhs_v])
        if rhs_v in new_match.vertex_map:
            context.merge_vertices(new_match.vertex_map[rhs_v], rhs_v1)
        else:
            new_match.vertex_map[rhs_v] = rhs_v1

    # Map the interior vertices to new vertices
    for vertex_id in rule.rhs.vertices.keys():
        if not rule.rhs.is_boundary(vertex_id):
            vertex = rule.rhs.vertices[vertex_id]
            new_vertex = context.add_vertex(deepcopy(vertex))
            new_match.vertex_map[vertex_id] = new_vertex
            new_match.vertex_image.add(new_vertex)

    # Now add the edges from rule.rhs to the context
    # and connect them using the vertex map
    for edge_id in rule.rhs.edges.keys():
        edge = rule.rhs.edges[edge_id]
        new_edge_id = context.add_edge(deepcopy(edge))
        new_edge = context.edges[new_edge_id]
        new_edge.sources = [new_match.vertex_map[v] for v in edge.sources]
        new_edge.targets = [new_match.vertex_map[v] for v in edge.targets]
        new_match.edge_map[edge_id] = new_edge_id
        new_match.edge_image.add(new_edge_id)

    return [new_match]
