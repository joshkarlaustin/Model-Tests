# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:15:46 2020

@author: Josh
"""

# Import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Coefficients for outcome relationships
betE = 1
bet2 = 1
bet3 = 1
bet4 = 1
bet5 = -1
bet6 = 1
bet7 = 1

# Centering of variables
muE = 0
mu2 = 0
mu3 = 0
mu4 = 0
mu5 = 0
mu6 = 0
mu7 = 0

# Standard deviation of variables
sdE = 3
sd2 = 1
sd3 = 1
sd4 = 1
sd5 = 1
sd6 = 1
sd7 = 1

# Bias terms (correlations between the variables and the residual)
bias2 = 0
bias3 = 0
bias4 = 0
bias5 = 0
bias6 = 0
bias7 = 0

# Colinearity terms
multi23 = 0.99
multi24 = 0
multi25 = 0
multi26 = 0
multi27 = 0
multi34 = 0
multi35 = 0
multi36 = 0
multi37 = 0
multi45 = 0.99
multi46 = 0
multi47 = 0
multi56 = 0
multi57 = 0
multi67 = 0

# Number of observations in the fake data
obs = 1000

# Create more complex constants including covariance matrix
mean = [muE, mu2, mu3, mu4, mu5, mu6, mu7]
variances = np.array([sdE, sd2, sd3, sd4, sd5, sd6, sd7])
varmat = np.outer(variances, np.transpose(variances))
cov = np.array([[1, bias2, bias3, bias4, bias5, bias6, bias7],
                [bias2, 1, multi23, multi24, multi25, multi26, multi27],
                [bias3, multi23, 1, multi34, multi35, multi36, multi37],
                [bias4, multi24, multi34, 1, multi45, multi46, multi47],
                [bias5, multi25, multi35, multi45, 1, multi56, multi57],
                [bias6, multi26, multi36, multi46, multi56, 1, multi67],
                [bias7, multi27, multi37, multi47, multi57, multi67, 1]])
varcov = varmat * cov
multiplier = [[betE], [bet2], [bet3], [bet4], [bet5], [bet6], [bet7]]

# Generate the fake data
rawdata = np.random.multivariate_normal(mean, cov, obs)
latent = rawdata @ multiplier
outcome = (latent > 0) * 1
X_data = rawdata[:, 1:]
combined = pd.DataFrame(np.append(outcome, X_data, axis=1))






