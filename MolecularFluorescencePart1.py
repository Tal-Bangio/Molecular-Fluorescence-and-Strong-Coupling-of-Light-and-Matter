import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Integral_Trapz_errors import TrapzWerr

def SbyCtoarr(path):
    # Creates an array of Areas (0), their errors (1), and corresponding concentrations (2) for specific file path
    Concentration = [0.0001, 0.0005, 0.0008, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
    S0c2 = np.zeros((10, 3))
    for i in range(0, 10):
        S0c2[i, 0:2] = TrapzWerr(path, "sample " + str(i + 1))
        S0c2[i, 2] = Concentration[i]
    return S0c2

colors = ["lime", "forestgreen", "mediumblue", "dodgerblue", "blueviolet", "purple", "mediumvioletred", "red", "darkorange", "gold"]
def plotting(path, name):
    Concentration = [0.0001, 0.0005, 0.0008, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
    for i in range(0, 10):
        df = pd.read_excel(path, sheet_name="sample " + str(i + 1))
        lambda0I2 = np.array(df)
        plt.plot(lambda0I2[0:, 0], lambda0I2[0:, 1], color=colors[i], label=str(Concentration[i]) + "mM")
    plt.title('Intensity by wavelength of ' + name)
    plt.xlabel("λ (nm)")
    plt.ylabel("I (A.U)")
    plt.legend(loc='upper right')
    plt.show()

fl_path = r"C:\Users\yuval\Documents\Files to open in Python\fl.xlsx"
plotting(fl_path, "fluorscein")
fl_S0c2 = SbyCtoarr(fl_path)

rodB_path = r"C:\Users\yuval\Documents\Files to open in Python\rodB.xlsx"
plotting(rodB_path, "Rhodamine B")
rodB_S0c2 = SbyCtoarr(rodB_path)

rod6G_path = r"C:\Users\yuval\Documents\Files to open in Python\rod6g.xlsx"
plotting(rod6G_path, "Rhodamine 6G")
rod6G_S0c2 = SbyCtoarr(rod6G_path)

print('Fluorescein', fl_S0c2)
print('Rhodamine B', rodB_S0c2)
print('Rhodamine 6G', rod6G_S0c2)

plt.plot(fl_S0c2[0:, 2], fl_S0c2[0:, 0], color="gold", label='Fluorescein')
plt.errorbar(fl_S0c2[0:, 2], fl_S0c2[0:, 0], yerr=fl_S0c2[0:, 1], xerr=fl_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.plot(rodB_S0c2[0:, 2], rodB_S0c2[0:, 0], color="deeppink", label='Rhodamine B')
plt.errorbar(rodB_S0c2[0:, 2], rodB_S0c2[0:, 0], yerr=rodB_S0c2[0:, 1], xerr=rodB_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.plot(rod6G_S0c2[0:, 2], rod6G_S0c2[0:, 0], color="darkorange", label='Rhodamine 6G')
plt.errorbar(rod6G_S0c2[0:, 2], rod6G_S0c2[0:, 0], yerr=rod6G_S0c2[0:, 1], xerr=rod6G_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.title('emission by concentration')
plt.xlabel("C (mM)")
plt.ylabel("S (A.U)")
plt.legend(loc='upper right')
plt.xlim([0, 0.12])
plt.ylim([0, 3.2])
plt.show()

plt.scatter(fl_S0c2[0:, 2], fl_S0c2[0:, 0], color="gold", label='Fluorescein', s=10)
plt.errorbar(fl_S0c2[0:, 2], fl_S0c2[0:, 0], yerr=fl_S0c2[0:, 1], xerr=fl_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.scatter(rodB_S0c2[0:, 2], rodB_S0c2[0:, 0], color="deeppink", label='Rhodamine B', s=10)
plt.errorbar(rodB_S0c2[0:, 2], rodB_S0c2[0:, 0], yerr=rodB_S0c2[0:, 1], xerr=rodB_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.scatter(rod6G_S0c2[0:, 2], rod6G_S0c2[0:, 0], color="darkorange", label='Rhodamine 6G', s=10)
plt.errorbar(rod6G_S0c2[0:, 2], rod6G_S0c2[0:, 0], yerr=rod6G_S0c2[0:, 1], xerr=rod6G_S0c2[0:, 2]/math.sqrt(12), color="b", fmt="None")
plt.title('emission by concentration')
plt.xlabel("C (mM)")
plt.ylabel("S (A.U)")
plt.legend(loc='upper right')
plt.show()

