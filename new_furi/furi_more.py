import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
#import opts
import time
import math

import os








def fu_one(No):
    No = No + 1
    def fu_one_(x):  # No=0,1,2
          # No=1,2,3
        x = 3.1415926535898 * x
        if (No % 2 == 0):
            y = torch.sin((No // 2) * x)
        elif (No % 2 == 1):
            y = torch.cos((No // 2) * x)
        else:
            print('wrong fu_one(No')
        return y
    return fu_one_

def fu_one_0(x):
    x = 3.1415926535898 * x
    return torch.cos(0 * x)
def fu_one_1(x):
    x = 3.1415926535898 * x
    return torch.cos(1 * x)
def fu_one_3(x):
    x = 3.1415926535898 * x
    return torch.cos(2 * x)
def fu_one_5(x):
    x = 3.1415926535898 * x
    return torch.cos(3 * x)
def fu_one_7(x):
    x = 3.1415926535898 * x
    return torch.cos(4 * x)
def fu_one_9(x):
    x = 3.1415926535898 * x
    return torch.cos(5 * x)
def fu_one_11(x):
    x = 3.1415926535898 * x
    return torch.cos(6 * x)
def fu_one_13(x):
    x = 3.1415926535898 * x
    return torch.cos(7 * x)
def fu_one_15(x):
    x = 3.1415926535898 * x
    return torch.cos(8 * x)
def fu_one_17(x):
    x = 3.1415926535898 * x
    return torch.cos(9 * x)
def fu_one_2(x):
    x = 3.1415926535898 * x
    return torch.sin(1 * x)
def fu_one_4(x):
    x = 3.1415926535898 * x
    return torch.sin(2 * x)
def fu_one_6(x):
    x = 3.1415926535898 * x
    return torch.sin(3 * x)
def fu_one_8(x):
    x = 3.1415926535898 * x
    return torch.sin(4 * x)
def fu_one_10(x):
    x = 3.1415926535898 * x
    return torch.sin(5 * x)
def fu_one_12(x):
    x = 3.1415926535898 * x
    return torch.sin(6 * x)
def fu_one_14(x):
    x = 3.1415926535898 * x

    return torch.sin(7 * x)
def fu_one_16(x):
    x = 3.1415926535898 * x
    return torch.sin(8 * x)
def fu_one_18(x):
    x = 3.1415926535898 * x
    return torch.sin(9 * x)
def fu_one_20(x):
    x = 3.1415926535898 * x
    return torch.sin(10 * x)
