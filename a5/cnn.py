#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1i
import torch
import torch.nn.utils
import torch.nn as nn
import unittest
import math


class CNN(nn.Module):
    def __init__(self, e_char: int, e_word: int, kernel_size: int =5):
        '''
        @param e_char (int): char embedding size.
        @param e_word (int): word embedding size.
        @param kernel_size (int): convlution kernel size.
        '''
        super(CNN, self).__init__()
        self.convlution_layer = nn.Conv1d(
            in_channels=e_char, out_channels=e_word, kernel_size=kernel_size)

    def forward(self, x_reshape: torch.Tensor) -> torch.Tensor:
        '''
        @param x_reshape (Tensor) : Tensor of size (batch_size, e_char, m_word).
        @return x_conv_out (Tensor): x_reshape after convlution, size (batch_size, e_word).
        '''
        x_conv = self.convlution_layer(x_reshape)
        x_conv_out, _ = torch.relu(x_conv).max(dim=2)
        return x_conv_out


class CNNCheck(unittest.TestCase):
    def check_shape(self):
        batch_size, e_char, e_word, m_word = 64, 20, 30, 25
        test_cnn = CNN(e_char=e_char, e_word=e_word)
        x_reshape = torch.rand((batch_size, e_char, m_word))
        x_conv_out = test_cnn.forward(x_reshape)

        self.assertEqual(x_conv_out.shape, (batch_size, e_word))

    def check_forward(self):
        batch_size, e_char, e_word, m_word = 64, 20, 30, 25
        test_cnn = CNN(e_char=e_char, e_word=e_word)
        test_cnn.convlution_layer.weight.data[:, :] = 0.0
        test_cnn.convlution_layer.bias.data[:] = 0.0

        x_reshape = torch.rand((batch_size, e_char, m_word))
        x_conv_out = test_cnn.forward(x_reshape)

        self.assertTrue(torch.allclose(x_conv_out, torch.zeros((batch_size, e_word))))

if __name__ == "__main__":
    unittest.TextTestRunner().run(CNNCheck("check_shape"))
    unittest.TextTestRunner().run(CNNCheck("check_forward"))
# END YOUR CODE
