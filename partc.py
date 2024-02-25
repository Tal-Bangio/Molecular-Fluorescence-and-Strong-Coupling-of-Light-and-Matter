import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.transform import downscale_local_mean
from scipy import ndimage
import openpyxl
#from google.colab import drive 
from scipy import signal
from scipy.signal import find_peaks
import plotly.graph_objects as go
from matplotlib import colormaps
import time

from time import sleep

import random




for theta in np.linspace(-40,40,41):
	if theta<0:
		df=pd.read_excel('Exciton polariton negative angles - Copy.xlsx', sheet_name=str(int(theta)))
	else:
		df=pd.read_excel('Exciton polariton - Copy.xlsx', sheet_name=str(int(theta)))
	x_cvs = ndimage.median_filter(df.iloc[:,0].values, size=20)
	y_cvs = ndimage.median_filter(df.iloc[:,1].values, size=20)
	plt.xlabel("λ[nm]")
	plt.ylabel("I[a.u]")
	plt.title("I(λ)")
	plt.plot(x_cvs,y_cvs, label=str(int(theta)))
	
plt.legend()
plt.show()


# theta>0
for theta in np.linspace(-40,40,11):
	if theta>-1:
		df=pd.read_excel('Exciton polariton - Copy.xlsx', sheet_name=str(int(theta)))
		x_cvs = ndimage.median_filter(df.iloc[:,0].values, size=20)
		y_cvs = ndimage.median_filter(df.iloc[:,1].values, size=20)
		plt.xlabel("λ[nm]")
		plt.ylabel("I[a.u]")
		plt.title("I(λ)")
		plt.plot(x_cvs,y_cvs, label=str(int(theta)), color=[random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)])
		time.sleep(0.1)
	
plt.legend()
plt.show()

# theta<0
for theta in np.linspace(-40,40,41):
	if theta<0:
		df=pd.read_excel('Exciton polariton negative angles - Copy.xlsx', sheet_name=str(int(theta)))
	else:
		break
	x_cvs = ndimage.median_filter(df.iloc[:,0].values, size=20)
	y_cvs = ndimage.median_filter(df.iloc[:,1].values, size=20)
	plt.xlabel("λ[nm]")
	plt.ylabel("I[a.u]")
	plt.title("I(λ)")
	plt.plot(x_cvs,y_cvs, label=str(int(theta)), color=[random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)])
	time.sleep(0.1)

	
plt.legend()
plt.show()




k_upper_list=[]
k_lower_list=[]
E_upper_list=[]
E_lower_list=[]
delta_k_upper_list=[]
delta_k_lower_list=[]
delta_E_upper_list=[]
delta_E_lower_list=[]


for theta in np.linspace(-40,40,41):
	#print(int(theta))
	if theta<0:
		df=pd.read_excel('Exciton polariton negative angles - Copy.xlsx', sheet_name=str(int(theta)))
	else:
		df=pd.read_excel('Exciton polariton - Copy.xlsx', sheet_name=str(int(theta)))
	# x_cvs = df.iloc[:,0].values
	# y_cvs = df.iloc[:,1].values
	x_cvs = ndimage.median_filter(df.iloc[:,0].values, size=30)
	y_cvs = ndimage.median_filter(df.iloc[:,1].values, size=30)



	indices = find_peaks(y_cvs, height=0.25,distance=80,width=10)[0]
	lst=list(y_cvs[indices])

	max_value = max(lst)
	max_index = list(y_cvs).index(max_value)

	sec_max_value_find=[a for i,a in enumerate(lst) if a<max_value]
	sec_max_value=max(sec_max_value_find)
	sec_max_index = list(y_cvs).index(sec_max_value)

	top=[max_index,sec_max_index]
	#calc
	delta_lambda=np.abs(x_cvs[top[0]]-x_cvs[top[1]])
	k_upper=1000*2*np.pi*np.sin(np.deg2rad(theta))/(x_cvs[min(top)])
	k_lower=1000*2*np.pi*np.sin(np.deg2rad(theta))/(x_cvs[max(top)])

	delta_k_upper=100*np.abs((0.034906585*3.46410161514)*2*np.pi*np.cos(np.deg2rad(theta))/(x_cvs[min(top)]))
	delta_k_lower=100*np.abs((0.034906585*3.46410161514)*2*np.pi*np.cos(np.deg2rad(theta))/(x_cvs[max(top)]))

	E_upper=1240/(x_cvs[min(top)])
	E_lower=1240/(x_cvs[max(top)])

	delta_E_upper=1000*(0.0001*3.46410161514)*1240/(x_cvs[min(top)])**2
	delta_E_lower=1000*(0.0001*3.46410161514)*1240/(x_cvs[max(top)])**2

	k_upper_list.append(k_upper)
	k_lower_list.append(k_lower)

	E_upper_list.append(E_upper)
	E_lower_list.append(E_lower)

	delta_k_upper_list.append(delta_k_upper)
	delta_k_lower_list.append(delta_k_lower)

	delta_E_upper_list.append(delta_E_upper)
	delta_E_lower_list.append(delta_E_lower)

	# plt.xlabel("λ[nm]")
	# plt.ylabel("I")
	# plt.title("I(λ)")
	# plt.plot(x_cvs,y_cvs)
	# plt.scatter(x_cvs[top],y_cvs[top])
	#plt.show()

#graph- upper and lower
x_ax=np.linspace(-8, 8, 1000)
#upper
def f(x):
	#a=[2.23,0.132,0,20.94,0.1]
	a=[1.579,0.03231,0,62.82,0.142]
	return a[0]/2+(a[1]*np.sqrt((x-a[2])**2+a[3]**2))/2+np.sqrt(4*a[4]**2+(a[0]-a[1]*np.sqrt((x-a[2])**2+a[3]**2))**2)
plt.plot(x_ax, f(x_ax), color='red')

#lower
def g(x):
	#a=[2.23,0.132,0,20.94,0.1]
	a=[2.2808,0.07573,0,30,0.03677]
	return a[0]/2+(a[1]*np.sqrt((x-a[2])**2+a[3]**2))/2-np.sqrt(4*a[4]**2+(a[0]-a[1]*np.sqrt((x-a[2])**2+a[3]**2))**2)
plt.plot(x_ax, g(x_ax), color='red')

# plt.scatter(k_upper_list,E_upper_list)
# plt.scatter(k_lower_list,E_lower_list)
plt.errorbar(k_upper_list,E_upper_list,yerr =delta_E_upper_list,xerr =delta_k_upper_list,fmt =' ',ecolor='b')
plt.errorbar(k_lower_list,E_lower_list,yerr =delta_E_lower_list,xerr =delta_k_lower_list,fmt =' ',ecolor='b')
plt.title("E(k)") 
plt.xlabel('k [1/μm]')
plt.ylabel('E [eV]')
plt.show()
	

#graph- upper and lower residuals
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('E(k)- residuals')
ax1.errorbar(k_upper_list,E_upper_list-f(np.asarray(k_upper_list, dtype=np.float32)),yerr =delta_E_upper_list,xerr =delta_k_upper_list,fmt =' ',ecolor='b')
ax1.plot(x_ax, 0*x_ax, color='red')
ax1.set_xlabel('k [1/μm]')
ax1.set_ylabel("$E_{upper i}-f(x_{i}) [eV]$")
ax1.autoscale(enable=True) 



ax2.errorbar(k_lower_list,E_lower_list-g(np.asarray(k_lower_list, dtype=np.float32)),yerr =delta_E_lower_list,xerr =delta_k_lower_list,fmt =' ',ecolor='b')
ax2.plot(x_ax, 0*x_ax, color='red')
ax2.set_xlabel('k [1/μm]')
ax2.set_ylabel("$E_{lower i}-f(x_{i}) [eV]$")
ax2.autoscale(enable=True) 


# plt.errorbar(k_upper_list,E_upper_list-f(np.asarray(k_upper_list, dtype=np.float32)),yerr =delta_E_upper_list,xerr =delta_k_upper_list,fmt =' ',ecolor='b')
# plt.plot(x_ax, 0*x_ax, color='red')
# plt.show()

# plt.errorbar(k_lower_list,E_lower_list-g(np.asarray(k_lower_list, dtype=np.float32)),yerr =delta_E_lower_list,xerr =delta_k_lower_list,fmt =' ',ecolor='b')
# plt.plot(x_ax, 0*x_ax, color='red')
plt.show()
print(min(np.abs(k_upper_list)))

# df=pd.read_excel('Exciton polariton negative angles - Copy.xlsx', sheet_name=str(int(theta)))
# x_cvs = df.iloc[:,0].values
# y_cvs = df.iloc[:,1].values



# indices = find_peaks(y_cvs, height=0.25,distance=100,width=10)[0]
# lst=list(y_cvs[indices])

# max_value = max(lst)
# max_index = list(y_cvs).index(max_value)

# sec_max_value_find=[a for i,a in enumerate(lst) if a<max_value]
# sec_max_value=max(sec_max_value_find)
# sec_max_index = list(y_cvs).index(sec_max_value)

# top=[max_index,sec_max_index]
# #calc
# delta_lambda=np.abs(x_cvs[top[0]]-x_cvs[top[1]])
# k_upper=2*np.pi*np.sin(theta)/(x_cvs[min(top)])
# k_lower=2*np.pi*np.sin(theta)/(x_cvs[max(top)])

# E_upper=1240/(x_cvs[min(top)])
# E_lower=1240/(x_cvs[max(top)])

# plt.xlabel("λ[nm]")
# plt.ylabel("I")
# plt.title("I(λ)")
# plt.plot(x_cvs,y_cvs)
# plt.scatter(x_cvs[top],y_cvs[top])
# plt.show()










