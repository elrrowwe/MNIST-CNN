"""
This file contains the base layer, from which all the layers inherit.
"""

class Layer:
    def __init__(self):
        self.inp = None
        self.out = None