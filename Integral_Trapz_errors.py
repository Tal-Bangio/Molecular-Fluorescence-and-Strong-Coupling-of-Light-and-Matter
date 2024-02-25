import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def noiseError(arr):  # Finds the error caused by noise. Using standard deviation (sqrt of variance)
    # Based on manufacturer info sheet, under 400nm should be only noise. Find that point.
    lambda400 = 0
    for i in range(0, len(arr[:, 0])):
        if arr[i, 0] > 400:
            lambda400 = i
            break
    noiseErr = math.sqrt((np.square(arr[:lambda400, 1])).sum()/(lambda400-1))
    return noiseErr

# Plan for calculating the area under the intensity(by wavelength) measurements. We'll use the trapezoid method.
# Find a way to take into consideration the errors. I won't incorporate the errors into the integral, but use the errors
# to evaluate the error of the final integral (based on the error of calculating the area of a trapezoid)
# Total error is the sqrt of the sum of the (error of each individual area calc)^2
# The files opened don't include the errors in them
def TrapzWerr(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    lambda0I2 = np.array(df)  # gives an array of the measurements of intensity and wave length with corresponding errors
    # Defines a matrix of the sum of two following intensity meas
    I_shift = np.zeros((len(lambda0I2[:, 0]) + 1, 1))
    I_shift[1:, 0] = lambda0I2[0:, 1]
    I_shift[0:len(lambda0I2[:, 0]), 0] = I_shift[0:len(lambda0I2[:, 0]), 0] + lambda0I2[0:, 1]
    I2sum = I_shift[1:len(lambda0I2[:, 0]), 0]
    # Defines a matrix of the difference in wavelength between two following meas
    lambda_shift = np.zeros((len(lambda0I2[:, 0]) + 1, 1))
    lambda_shift[1:, 0] = -lambda0I2[0:, 0]
    lambda_shift[0:len(lambda0I2[:, 0]), 0] = lambda_shift[0:len(lambda0I2[:, 0]), 0] + lambda0I2[0:, 0]
    lambda2diff = lambda_shift[1:len(lambda0I2[:, 0]), 0]
    # transpose the λ0 part of the matrix
    lambda_trans = lambda2diff.transpose()
    # multiply (np.matmul) the transposed λ0 with the matrix I2sum (holds sums of two current and next I meas)
    Area = 0.5 * np.matmul(lambda_trans, I2sum)
    print(Area)
    # For error calc, two new matrices will also be needed. Errors calculated using
    lambda_err = np.full((len(lambda2diff)), abs(lambda0I2[1500, 0] - lambda0I2[1501, 0]) / math.sqrt(12))
    I_err = np.full((len(I2sum)), noiseError(lambda0I2))
    Err_Area_sqr = 0.5 * (np.square(lambda2diff) * np.square(I_err) + np.square(I2sum) * np.square(lambda_err))
    Area_err_tot = math.sqrt(Err_Area_sqr.sum())
    print(Area_err_tot)
    return [Area, Area_err_tot]
