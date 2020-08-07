import numpy
import scipy.special
from collections import deque
import random


class NeuralNetwork:
    """
    General Neural network class.
    """

    def __init__(self, deptharray, lr, bias=False):
        """
        General NN of size n where n >= 2. if bias = True then bias nodes are added at every layer except output.
        """
        if bias:
            self.bias = 1
        else:
            self.bias = 0
        self.depth = len(deptharray)
        self.deptharray = deptharray
        self.i = deptharray[0]
        self.o = deptharray[-1]
        self.lr = lr
        self.W = []

        i = 0
        while i < len(deptharray) - 1:
            self.W.append(numpy.random.normal(0, pow(deptharray[i + 1], -.5), (deptharray[i + 1], deptharray[i] + self.bias)))
            i += 1

        #self.W_ih = numpy.random.normal(0, pow(self.h1, -.5), (self.h1, self.i + 1))
        #self.W_hh = numpy.random.normal(0, pow(self.h2, -.5), (self.h2, self.h1 + 1))
        #self.W_ho = numpy.random.normal(0, pow(self.o, -.5), (self.o, self.h2 + 1))

    """
    def train(self, inputs, targets):
        ins = inputs.copy()
        ins = self.norm(ins)
        ins.append(self.bias)
        target = numpy.array(targets, ndmin=2).T
        ins = numpy.array(ins, ndmin=2).T

        h1_ins = numpy.dot(self.W_ih, ins)

        h1_outs_nb = self.activation(h1_ins) #activation of hidden layer 1 with no bias node obviously duhhh.

        h1_outs = numpy.insert(h1_outs_nb, h1_outs_nb.size, 1, 0) #activation of hidden layer 2

        h2_ins = numpy.dot(self.W_hh, h1_outs)

        h2_outs_nb = self.activation(h2_ins)

        h2_outs = numpy.insert(h2_outs_nb, h2_outs_nb.size, 1, 0)

        final_outs_in = numpy.dot(self.W_ho, h2_outs)
        final_out = self.activation(final_outs_in)
        error = target - final_out

        hidden2_errors = numpy.dot(self.W_ho.T, error)
        backprop_errorH2 = numpy.delete(hidden2_errors, -1, 0) #removing bias errors so they dont get sent back to input
        hidden1_errors = numpy.dot(self.W_hh.T, backprop_errorH2)
        backprop_errorH1 = numpy.delete(hidden1_errors, -1, 0) #removing bias errors
        self.W_ho += self.lr * numpy.dot((error * final_out * (1 - final_out)), numpy.transpose(numpy.insert(h2_ins, h2_ins.size, 1, 0)))
        self.W_hh += self.lr * numpy.dot((backprop_errorH2 * h2_outs_nb * (1 - h2_outs_nb)), numpy.transpose(numpy.insert(h1_ins, h1_ins.size, 1, 0)))
        self.W_ih += self.lr * numpy.dot((backprop_errorH1 * h1_outs_nb * (1 - h1_outs_nb)), numpy.transpose(inputs))
    """

    def train(self, inputs, targets):
        """
        Error due to bias node when adjusting weights. possibly break this ugly ass function up too.
        """
        ins = self.norm(inputs)
        layer_vals_o = deque()
        layer_vals_i = deque()
        layer_vals_i.append(numpy.array(ins, ndmin=2).T)
        if self.bias == 1:
            ins.append(self.bias)
        ins = numpy.array(ins, ndmin=2).T
        layer_vals_o.append(ins)

        ind = 1
        while ind < self.depth - 1:
            l_ins = numpy.dot(self.W[ind], ins)
            layer_vals_i.append(l_ins)
            l_outs = self.activation(l_ins)

            if ind + 2 == self.depth:
                pass
            else:
                if self.bias == 1:
                    l_outs = numpy.insert(l_outs, l_outs.size, self.bias, 0)
                ins = l_outs.copy()
            layer_vals_o.append(l_outs)
            ind += 1

        target = numpy.array(targets, ndmin=2).T
        BPerrors = deque()
        error = target - l_outs
        BPerrors.appendleft(error)
        ind = self.depth - 2
        while ind >= 0:     #populate Backpropogation error queue
            curr_error = numpy.dot(self.W[ind].T, error)
            BPerrors.appendleft(numpy.delete(curr_error, -1, 0))
            error = curr_error
            ind -= 1

        ind = self.depth - 2
        layer_vals_i.pop()
        while len(layer_vals_i) != 0:
            error = BPerrors.pop()
            nb_outs = layer_vals_o.pop()
            nb_ins = layer_vals_i.pop()
            if self.bias == 0:
                self.W[ind] += self.lr * numpy.dot((error * nb_outs * (1 - nb_outs)), numpy.transpose(nb_ins)) #this only works for non bias cases. not even actually.
            else:
                self.W[ind] += self.lr * numpy.dot((error * nb_outs * (1 - nb_outs)), numpy.transpose(numpy.insert(nb_ins, nb_ins.size, 1, 0)))
            ind -= 1



    #      REFERENCE FOR NEW QUERY() FUNCTION.
    #    ins = inputs_list.copy()
    #    ins = self.norm(ins)      #remove if you want to customize your normalization function
    #    ins.append(self.bias)
    #    ins = numpy.array(ins, ndmin=2).T
    #    hidden1_inputs = numpy.dot(self.W_ih, ins)
    #    hidden1_outputs = self.activation(hidden1_inputs)
    #    hidden1_outputs = numpy.insert(hidden1_outputs, hidden1_outputs.size, 1, 0)
    #    hidden2_inputs = numpy.dot(self.W_hh, hidden1_outputs)
    #    hidden2_outputs = self.activation(hidden2_inputs)
    #    hidden2_outputs = numpy.insert(hidden2_outputs, hidden2_outputs.size, 1, 0)
    #    output_inputs = numpy.dot(self.W_ho, hidden2_outputs)
    #    outputs = self.activation(output_inputs)
    #    return self.norm(outputs)

    def query(self, inputs):
        """
        Query function for NN. Loops through weight matrices till output reached.
        """
        ins = self.norm(inputs)
        if self.bias == 1:
            ins.append(self.bias)
        ins = numpy.array(ins, ndmin=2).T

        ind = 1
        while ind < self.depth - 1:
            l_ins = numpy.dot(self.W[ind], ins)
            l_outs = self.activation(l_ins)
            if ind + 2 == self.depth:
                return list(l_outs.flatten())
            else:
                if self.bias == 1:
                    l_outs = numpy.insert(l_outs, l_outs.size, self.bias, 0)
                ins = l_outs.copy()
            ind += 1

    def norm(self, outs):
        x = sum(outs)
        for i in range(len(outs)):
            outs[i] = outs[i] / x
        return outs

    def activation(self, x):
        return scipy.special.expit(x)

    def mutate(self, rate):
        """
        Useful for Genetic algorithms.
        """

        for i in range(len(self.W_ih)):
            for j in range(len(self.W_ih[i])):
                if random.uniform(0, 1) < rate:
                    self.W_ih[i][j] = random.uniform(-1, 1)

        for i in range(len(self.W_hh)):
            for j in range(len(self.W_hh[i])):
                if random.uniform(0, 1) < rate:
                    self.W_ih[i][j] = random.uniform(-1, 1)

        for i in range(len(self.W_ho)):
            for j in range(len(self.W_ho[i])):
                if random.uniform(0, 1) < rate:
                    self.W_ih[i][j] = random.uniform(-1, 1)









