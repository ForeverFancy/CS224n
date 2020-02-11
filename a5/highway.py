#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1h
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils
import torch.nn as nn
import unittest
import math


class Highway(nn.Module):
    def __init__(self, e_word: int):
        '''
        @param e_word (int): the input feature size and also the output feature size.
        '''
        super(Highway, self).__init__()
        self.project_layer = nn.Linear(e_word, e_word, bias=True)
        self.gate = nn.Linear(e_word, e_word, bias=True)

    def forword(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        '''
        @param x_conv_out(tensor): the output of convolution network.
        @return x_highway(Tensor): output of highway layer.
        '''
        x_proj = torch.relu(self.project_layer(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway


class HighwayCheck(unittest.TestCase):

    def check_shape(self):
        batch_size, e_word = 64, 30
        highway = Highway(e_word)

        x_conv_out = torch.randn((batch_size, e_word))
        x_highway = highway.forword(x_conv_out)

        self.assertEqual(x_highway.shape, x_conv_out.shape)
        self.assertEqual(x_highway.shape, (batch_size, e_word))

    def check_forward(self):
        batch_size, e_word = 64, 30
        conv_highway = Highway(e_word)
        conv_highway.gate.weight.data[:, :] = 0.0
        conv_highway.gate.bias.data[:] = -math.inf

        x_conv_out = torch.randn((batch_size, e_word))
        x_highway = conv_highway.forword(x_conv_out)

        self.assertTrue(torch.allclose(x_conv_out, x_highway))

        proj_highway = Highway(e_word)
        proj_highway.gate.weight.data[:, :] = 0.0
        proj_highway.gate.bias.data[:] = +math.inf
        proj_highway.project_layer.weight.data[:, :] = torch.eye(e_word)
        proj_highway.project_layer.bias.data[:] = 0.0

        x_conv_out = torch.randn((batch_size, e_word))
        x_highway = conv_highway.forword(x_conv_out)

        self.assertTrue(torch.allclose(x_conv_out, x_highway))


if __name__ == "__main__":
    unittest.TextTestRunner().run(HighwayCheck("check_shape"))
    unittest.TextTestRunner().run(HighwayCheck("check_forward"))

# END YOUR CODE
