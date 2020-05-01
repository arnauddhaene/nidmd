# This Python file uses the following encoding: utf-8
import numpy as np
from utils import *


class Mode:

    def __init__(self, value, vector, order):
        """
        Constructor

        :param value: eigenvalue
        :param vector: eigenvector
        :param order: order
        """

        self.order = order

        if type(value) in [list, tuple] and len(value) == 2:
            self.is_complex_conjugate = True
            self.value = value[0]
        elif type(value) == np.complex128:
            self.is_complex_conjugate = False
            self.value = np.real(value)

        self.intensity = vector

        # TODO: modify sampling time with user input
        self.damping_time = (-1 / np.log(np.abs(self.value))) * 0.72
        if self.is_complex_conjugate:
            self.period = ((2 * PI) / np.abs(np.angle(self.value))) * 0.72
        else:
            self.period = np.inf

    def __repr__(self):
        return "Mode()"

    def __str__(self):
        return "Mode {}".format(self.order)

    def print(self):
        return "Mode {0} — ∆ = {1} — T = {2}".format(self.order, '%s' % float('%.2f' % self.damping_time), '%s' % float('%.2f' % self.period))

    def print_value(self):
        if self.is_complex_conjugate:
            return '%s' % float('%.2f' % np.real(self.value)) + ' ' + u"\u00B1" + ' ' + '%s' % float('%.2f' % np.imag(self.value)) + 'j'
        else:
            return '%s' % float('%.2f' % self.value)
