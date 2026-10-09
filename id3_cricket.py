"""Build and use an ID3 decision tree for the cricket dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
from math import log2
from typing import Union


DATASET = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Cricket": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play Cricket": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Cricket": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Cricket": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Cricket": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Cricket": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Cricket": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Cricket": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Cricket": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play Cricket": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play Cricket": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play Cricket": "Yes"},
]

ATTRIBUTES = ["Outlook", "Temperature", "Humidity", "Wind"]
TARGET = "Play Cricket"


@dataclass
class Leaf:
    label: str


@dataclass
class Decision:
    attribute: str
    branches: dict[str, "Tree"]


Tree = Union[Leaf, Decision]


def entropy(rows: list[dict[str, str]]) -> float:
    """Return target entropy for a group of rows."""
    counts = Counter(row[TARGET] for row in rows)
    total = len(rows)
    return -sum((count / total) * log2(count / total) for count in counts.values())


def information_gain(rows: list[dict[str, str]], attribute: str) -> float:
    """Return gain from splitting rows on one attribute."""
    total = len(rows)
    remainder = sum(
        (len(group) / total) * entropy(group)
        for value in sorted({row[attribute] for row in rows})
        for group in [[row for row in rows if row[attribute] == value]]
    )
    return entropy(rows) - remainder


def majority_class(rows: list[dict[str, str]]) -> str:
    return Counter(row[TARGET] for row in rows).most_common(1)[0][0]


def build_tree(rows: list[dict[str, str]], attributes: list[str]) -> Tree:
    labels = {row[TARGET] for row in rows}
    if len(labels) == 1:
        return Leaf(next(iter(labels)))
    if not attributes:
        return Leaf(majority_class(rows))

    gains = {attribute: information_gain(rows, attribute) for attribute in attributes}
    best_attribute = max(attributes, key=lambda attribute: gains[attribute])
    remaining = [attribute for attribute in attributes if attribute != best_attribute]
    branches: dict[str, Tree] = {}

    for value in sorted({row[best_attribute] for row in rows}):
        subset = [row for row in rows if row[best_attribute] == value]
        branches[value] = build_tree(subset, remaining)
    return Decision(best_attribute, branches)


def print_tree(tree: Tree, indent: str = "") -> None:
    if isinstance(tree, Leaf):
        print(f"{indent}-> {tree.label}")
        return
    for value, branch in tree.branches.items():
        print(f"{indent}{tree.attribute} = {value}")
        print_tree(branch, indent + "  ")


def predict(tree: Tree, sample: dict[str, str]) -> str:
    while isinstance(tree, Decision):
        tree = tree.branches[sample[tree.attribute]]
    return tree.label


def save_tree_image(tree: Tree, filename: str = "decision_tree.svg") -> None:
    positions: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int, str]] = []
    leaf_number = 0
    max_depth = 0

    def layout(node: Tree, depth: int) -> float:
        nonlocal leaf_number, max_depth
        max_depth = max(max_depth, depth)
        if isinstance(node, Leaf):
            x = 140 + leaf_number * 180
            leaf_number += 1
        else:
            child_positions = []
            for value, child in node.branches.items():
                child_x = layout(child, depth + 1)
                child_positions.append(child_x)
                edges.append((id(node), id(child), value))
            x = sum(child_positions) / len(child_positions)
        positions[id(node)] = (x, 90 + depth * 130)
        return x

    layout(tree, 0)
    width = max(400, int(leaf_number * 180 + 100))
    height = (max_depth + 1) * 130 + 70
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="20" y="30" font-family="Arial, sans-serif" font-size="20" '
        'font-weight="bold" fill="#172033">ID3 Decision Tree: Play Cricket</text>',
    ]

    for parent_id, child_id, value in edges:
        parent_x, parent_y = positions[parent_id]
        child_x, child_y = positions[child_id]
        label_x = (parent_x + child_x) / 2
        label_y = (parent_y + child_y) / 2 - 5
        elements.append(
            f'<line x1="{parent_x}" y1="{parent_y + 24}" x2="{child_x}" '
            f'y2="{child_y - 24}" stroke="#64748b" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="13" fill="#334155">'
            f'{escape(value)}</text>'
        )

    def draw_nodes(node: Tree) -> None:
        x, y = positions[id(node)]
        if isinstance(node, Leaf):
            fill = "#dcfce7" if node.label == "Yes" else "#fee2e2"
            elements.append(
                f'<rect x="{x - 48}" y="{y - 24}" width="96" height="48" rx="10" '
                f'fill="{fill}" stroke="#334155" stroke-width="2"/>'
            )
            text = f"Play: {node.label}"
        else:
            elements.append(
                f'<rect x="{x - 65}" y="{y - 24}" width="130" height="48" rx="10" '
                'fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>'
            )
            text = node.attribute
            for child in node.branches.values():
                draw_nodes(child)
        elements.append(
            f'<text x="{x}" y="{y + 6}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="14" font-weight="bold" '
            f'fill="#172033">{escape(text)}</text>'
        )

    draw_nodes(tree)
    elements.append("</svg>")
    with open(filename, "w", encoding="utf-8") as image_file:
        image_file.write("\n".join(elements))


def main() -> None:
    print(f"Dataset entropy: {entropy(DATASET):.3f}")
    print("Information gain at the root:")
    for attribute in ATTRIBUTES:
        print(f"  {attribute}: {information_gain(DATASET, attribute):.3f}")

    tree = build_tree(DATASET, ATTRIBUTES)
    print("\nDecision tree:")
    print_tree(tree)
    save_tree_image(tree)
    print("\nDecision tree image saved to: decision_tree.svg")

    sample = {
        "Outlook": "Sunny",
        "Temperature": "Cool",
        "Humidity": "High",
        "Wind": "Strong",
    }
    print(f"\nNew sample: {sample}")
    print(f"Predicted class: {predict(tree, sample)}")


if __name__ == "__main__":
    main()
