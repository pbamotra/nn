'''Implementation of neural network from Andrej Karpathy blog in Python.

Source: http://karpathy.github.io/neuralnets/#example-single-neuron
'''
import math
from collections import namedtuple


class Unit:
    '''Represents all the wirings of the neural network.'''

    def __init__(self, value, grad):
        '''Initialize the unit.

        Args:
            value - input from the parent*
            grad - local gradient
        '''
        self.value = value
        self.grad = grad

class MultiplyGate:
    '''Implement a multiplication gate.'''

    def forward(self, u0, u1):
        '''Implement forward iteration.

        Args:
            u0 - first input unit
            u1 - second input unit
        '''
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)
        return self.utop

    def backward(self):
        '''Implement backward iteration.'''
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad

class AddGate:
    def forward(self, u0, u1):
        '''Implement forward iteration.

        Args:
            u0 - first input unit
            u1 - second input unit
        '''
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(self.u0.value + self.u1.value, 0.0)
        return self.utop

    def backward(self):
        '''Implement backward iteration.'''
        self.u0.grad += 1.0 * self.utop.grad
        self.u1.grad += 1.0 * self.utop.grad

class SigmoidGate:
    @staticmethod
    def sig(arg):
        '''
        Implement sigmoid function.

        Args:
            arg: input to the sigmoid activation function
        '''
        return 1.0 / (1 + math.exp(-arg))

    def forward(self, u0):
        '''Implement forward iteration.

        Args:
            u0 - input unit
        '''
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        '''Implement backward iteration.'''
        s = self.sig(self.u0.value)
        derivative_term = s * (1 - s)
        self.u0.grad += derivative_term * self.utop.grad


if __name__ == '__main__':
    # create input units
    a = Unit(1.0, 0.0)
    b = Unit(2.0, 0.0)
    c = Unit(-3.0, 0.0)
    x = Unit(-1.0, 0.0)
    y = Unit(3.0, 0.0)

    # create different gates/activations!
    mulg0 = MultiplyGate()
    mulg1 = MultiplyGate()
    addg0 = AddGate()
    addg1 = AddGate()
    sg0 = SigmoidGate()

    # do forward iteration
    def forwardNeuron():
        ax = mulg0.forward(a, x)  # a*x = -1
        by = mulg1.forward(b, y)  # b*y = 6
        axpby = addg0.forward(ax, by)  # a*x + b*y = 5
        axpbypc = addg1.forward(axpby, c)  # a*x + b*y + c = 2
        s = sg0.forward(axpbypc)  # sig(a*x + b*y + c) = 0.8808
        return s

    s = forwardNeuron()
    # print output of the network
    print 'circuit output: ', round(s.value, 4)

    s.grad = 1.0;
    sg0.backward()  # writes gradient into axpbypc
    addg1.backward()  # writes gradients into axpby and c
    addg0.backward()  # writes gradients into ax and by
    mulg1.backward()  # writes gradients into b and y
    mulg0.backward()  # writes gradients into a and x

    step_size = 0.01
    a.value += step_size * a.grad  # a.grad is -0.105
    b.value += step_size * b.grad  # b.grad is 0.315
    c.value += step_size * c.grad  # c.grad is 0.105
    x.value += step_size * x.grad  # x.grad is 0.105
    y.value += step_size * y.grad  # y.grad is 0.210

    s = forwardNeuron()
    # print output of the network
    print 'circuit output: ', round(s.value, 4)
