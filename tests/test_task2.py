# from collections import OrderedDict
# import pandas as pd
# import numpy as np
# import os

from pytest import fixture


@fixture(scope='module')
def wind_predictor():
    import wind_predictor
    return wind_predictor


# Check data has imported correctly


# check the input data is correct


# Check the dataloader is working correctly


# Check the input image size is correct


# Check the output image size is correct
