#!/usr/bin/env python3
"""Génération de palette déterministe de couleurs, reprise telle quelle du pipeline existant"""

import math
import random


class paletteGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def _dist(self, c1, c2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    def _is_grey(self, color, threshold=50):
        return (max(color) - min(color)) < threshold

    def add_color(self, color=None):
        if color:
            self.colors.append(color)
            return color

        random.seed(self.seed + len(self.colors))
        best_color = None
        max_dist_min = -1

        for _ in range(2000):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if self._is_grey(color):
                continue
            dist_min = min(self._dist(color, c) for c in self.colors)
            if dist_min > max_dist_min:
                max_dist_min = dist_min
                best_color = color

        if best_color is None:
            best_color = color

        self.colors.append(best_color)
        return best_color

    def generate_palette(self, n):
        if n < 0:
            return None
        if n <= len(self.colors):
            return self.colors[0:n]

        missing = n - len(self.colors)
        for _ in range(missing):
            self.add_color()
        return self.colors

    def index(self, color):
        if color in self.colors:
            return self.colors.index(color)
        return None

    def at(self, idx):
        if idx >= 0 and idx < len(self.colors):
            return list(self.colors[idx])
        return None