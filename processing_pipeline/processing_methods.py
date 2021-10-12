import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
from patsy import dmatrices

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_blobs

class scatters:
    def __init__(self):
        pass

    def smoothed_Scatter(self, data, X, Y, splits=5, x_method=np.median, y_method=np.mean):
        '''
        this method takes X and Y points and creates a smoothed scatter plot.
        splits defines the amount of splits in the data
        x_method defines the method to determine the x point
        y_method defines the method to determine the y point
        '''

        divisors = (np.max(data[X]) - np.min(data[X]))/splits
        x_smooth, y_smooth = [], []
        for k in range(len(splits)):
            top, bot = np.min(data[X])+(k+1)*divisors, np.min(data[X])+k*(divisors)
            cond = data[(data[X] >= bot) & (data[X] <= top)]
            Y_pts = np.mean(cond[Y])
            x_smooth.append((bot + top) / 2)
            y_smooth.append(Y_pts)
        return (x_smooth, y_smooth)


