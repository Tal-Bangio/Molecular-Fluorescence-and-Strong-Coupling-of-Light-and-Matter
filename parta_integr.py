# load_mat_file

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
from scipy import stats
from skimage import img_as_float
import cv2
import sys
import random as rng
import os
import os.path as osp
from eddington import FittingData, fit, fitting_functions_list
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import simpledialog
from scipy import ndimage
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)

PIC_START = 900
PIC_END = 3978
PIC_HIGHT_START = 1266
PIC_HIGHT_END = 1570
PIC_SIZE_X = 4288
SIZE_IN_PIXELS = 3596


def FindAllxOnΔx (Av, sigma):
	Δw = PIC_HIGHT_END - PIC_HIGHT_START
	AvErr = np.zeros (len (Av))
	for x in range (len (Av)):
		Δx = int (Δw * np.tan ((x + PIC_START - (PIC_SIZE_X/2)) / ((PIC_SIZE_X/2)/np.tan (10 * np.pi/180))))
		#print ((x + PIC_START - (PIC_SIZE_X/2)) / ((PIC_SIZE_X/2)/np.tan (10 * np.pi/180)))
		if (Δx > 0):
			AvErr [x] = np.sqrt ((stats.tstd (Av [x:x + Δx])**2)/abs (Δx) + ((2**(-12))**2)/12 + sigma [x])
		elif (Δx < 0):
			AvErr [x] = np.sqrt ((stats.tstd (Av [x + Δx:x])**2)/abs (Δx) + ((2**(-12))**2)/12 + sigma [x])
		else:
			AvErr [x] = np.sqrt (((2**(-16))**2)/12 + sigma [x])
	#print (AvErr)
	return AvErr
	

def get_files_byformat (dir) :
	onlyfiles = [f for f in os.listdir(dir) if osp.isfile(osp.join(dir, f))]
	obj = []
	print (onlyfiles)
	for file in onlyfiles:
		if (((file.find (".tiff") == len(file) - 4) or (file.find (".TIFF") == len(file) - 4) or (file.find (".tif") == len(file) - 4) or (file.find (".TIF") == len(file) - 4))):
			obj.append (file)
	return obj

def ImageAnalysis (dir, image):
    #load the mat files  
    # Load the input image
	Data = {}
	print (image)
	img = cv2.imread(dir + "\\" + image, cv2.IMREAD_ANYDEPTH)
	img_gs = cv2.imread(dir + "\\" + image, cv2.IMREAD_ANYDEPTH) 
	
	# Obtain the dimensions of the image array 
	# using the shape method 
		
	A1 = img_as_float(img_gs)
	Av = A1 [PIC_HIGHT_START, PIC_START:PIC_END]  # Indexing in Python is 0-based
	for i in range (PIC_HIGHT_END - PIC_HIGHT_START):
		Av += A1 [PIC_HIGHT_START + i, PIC_START:PIC_END] / (PIC_HIGHT_END - PIC_HIGHT_START)
	
	sigma1_list = []
	for i in range (PIC_END - PIC_START):
		if (((stats.tstd (A1 [PIC_HIGHT_START : PIC_HIGHT_END, i])**2)/(PIC_HIGHT_END - PIC_HIGHT_START)) > Av [i]/10):
			sigma1_list.append (Av [i]/10)
		else:
			sigma1_list.append ((stats.tstd (A1 [PIC_HIGHT_START : PIC_HIGHT_END, i])**2)/ (PIC_HIGHT_END - PIC_HIGHT_START))
		#print ((stats.tstd (A1 [PIC_HIGHT_START : PIC_HIGHT_END, i])**2)/(PIC_HIGHT_END - PIC_HIGHT_START))


	sigma_list = []
	for i in range (PIC_END - PIC_START):
		if (i < 50):
			sigma_list.append ((stats.tstd (Av [0:i + 50])**2)/(i + 50))
		elif (i > (PIC_END - 50)):
			sigma_list.append ((stats.tstd (Av [i - 50:PIC_END])**2)/(PIC_END - i + 50))
		else:
			sigma_list.append ((stats.tstd (Av [i - 50:i + 50])**2)/100)
	sigma = np.array (sigma_list) + np.average (sigma_list [:100])
	Av = ndimage.median_filter(Av, size=100)
	x = np.linspace(0, 10, len(Av))
	end = len (Av) - 1
	for i in range (len (Av)):
		if (Av [i] < 0.01):
			end = i
			break
		
	if (end == 0):
		end = len (Av)
	Avl = np.log(Av [:end])
	Data ["Signal"] = Av [:end]
	Data ["SignalErr"] = FindAllxOnΔx (Av [:end], sigma [:end])
	Data ["Log"] = Avl
	Data ["LogErr"] = (FindAllxOnΔx (Av [:end], sigma[:end])*0 + (FindAllxOnΔx (Av [:end], sigma[:end])/np.average (Av [:end])) [int (len (Data ["Log"])/2)])
	#Data ["LogErr"] = FindAllxOnΔx (Av [:end], sigma[:end])/Av [:end]
	#print (Data ["LogErr"])
	pix_to_mm = 7.5/3596
	Data ["x"] = np.linspace(0, 7.5, PIC_SIZE_X) [:len(Data ["Signal"])]
	Data ["xErr"] = (np.linspace(0, 7.5, len(Data ["Signal"])) * 0 + 1) * pix_to_mm/np.sqrt (12)

	try:
		lum = 0

		for i in range (len (Data ["Signal"])):
			if (Data ["x"] [i] > 1.25):
				break
			lum += ((Data ["Signal"] [i] + Data ["Signal"] [i + 1])/2) * (Data ["x"] [i + 1] - Data ["x"] [i])

		print ("Luminosity: " + str (lum))
	except IndexError:
		return image, Data, img_gs
	#print (len (Data ["Log"]), len (Data ["LogErr"]), len (Data ["x"]), len (Data ["xErr"]))

	return image, Data, img_gs

def linear (x, a):
	return (a [1] * x + a [0])

def HighPoly (a, n, x):
        return sum([a[i] * x ** i for i in range(n + 1)])

def IdealPolynom (data):
	x = data ["x"]
	y = data ["Log"]
	#plt.scatter(T,V)
	#plt.show()
	#T=x, V=y

	max_D = 20 #this is the maximal degree for the polynomial
	k_fold = 20 #number of splits
	train_err = np.zeros(max_D)
	test_err = np.zeros(max_D)
	for i in range(max_D):
		test_err_curr = 0 
		train_err_curr = 0
		for j in range(k_fold):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
			p = np.poly1d(np.polyfit(x_train, y_train, i + 1)) #find the polynomial of degree i 
			y_train_hat = p(x_train) #the model output for the train data
			y_test_hat = p(x_test) #the model output for the test data
			train_err_curr += np.dot(y_train_hat - y_train, y_train_hat - y_train) / x_train.shape[0]
			test_err_curr += np.dot(y_test_hat - y_test, y_test_hat - y_test) / x_test.shape[0]
		train_err[i] = train_err_curr / k_fold 
		test_err[i] = test_err_curr / k_fold
	return max_D, test_err, train_err
	
	
images = get_files_byformat (sys.argv [1])
table = open (sys.argv [1] + "\\fitting_data.csv", "w")
table.write (",linear parameters,chi squered,6 od poly,chi squered\n")
for image in images:
	name, data, image_data = ImageAnalysis (sys.argv [1], image)
	pix_to_mm = 7.5/SIZE_IN_PIXELS
	x = np.linspace(0, 7.5, PIC_SIZE_X) [:len(data ["Signal"])]
	fitting_data = FittingData (data)
	fitting_data.x_column = "x"
	fitting_data.xerr_column = "xErr"
	fitting_data.y_column = "Log"
	fitting_data.yerr_column = "LogErr"
	fitting_result = fit(fitting_data, fitting_functions_list.linear)  # Do the actual fitting
	
	print(fitting_result)  # Print the results
	max_D, test_err, train_err = IdealPolynom (data)
	#print (np.where(abs (test_err - test_err) == abs (min (test_err - test_err))) [0][0] + 1)
	diff = abs (test_err - train_err)
	#print (np.where(diff == min (diff)) [0] [0] + 1)
	nameNoExt = name [:name.rfind (".")]
	table.write (nameNoExt + ",\"")
	for i, ai in enumerate (fitting_result.a):
		#print ("a[" + str (i) + "] = " + str (ai) + " \u00B1 " + str (fitting_result.aerr [i]) + " (" + str (fitting_result.arerr [i]) + "%)\n")
		table.write ("a[" + str (i) + "] = " + str (ai) + " \u00B1 " + str (fitting_result.aerr [i]) + " (" + str (fitting_result.arerr [i]) + "%)")
		if (i < len (fitting_result.a) - 1):
			table.write ("\n")
	table.write ("\"," + str (fitting_result.chi2_reduced) + ",\"")
	try:
		os.mkdir (sys.argv [1] + "\\" + nameNoExt)
	except OSError:
		print (sys.argv [1] + "\\" + nameNoExt + " Already exist !")
	FitFile = open (sys.argv [1] + "\\" + nameNoExt + "\\" + "FitFile.txt", "w")
	FitFile.write ("Linear Fit: \n\n")
	FitFile.write (str (fitting_result) + "\n\n")
	
	plt.clf()
	plt.imshow(image_data, cmap="twilight_shifted")
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\" + "ImageMask.png")
	plt.figure()
	#plt.plot(x, data ["Signal"])
	#plt.errorbar(x, data ["Signal"], yerr=data ["SignalErr"], xerr=data ["xErr"], color="b")
	plt.scatter(x, data ["Signal"], s=1)
	plt.xlabel('x[cm]')
	plt.ylabel(name + ' - Power [AU]')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\" + "RawData.png", dpi=1000)
	#plt.show ()

	plt.clf()
	#plt.errorbar(x, data ["Log"], yerr=data ["LogErr"], xerr=data ["xErr"], color="b")
	plt.scatter(x, data ["Log"], s=1)
	plt.plot(x, linear (x, fitting_result.a), color = "orange")
	plt.legend(["log(Av)", "6th Order Poly"], loc="upper right")
	plt.xlabel('x[cm]')
	plt.ylabel(name + ' - Log (I) [au]')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\" + "LogAndFit.png")

	plt.clf()
	#plt.plot(x, data ["Log"] - linear (x, fitting_result.a))
	#plt.errorbar(x, data ["Log"] - linear (x, fitting_result.a), yerr=data ["LogErr"], xerr=data ["xErr"], color="b", capsize=3)
	plt.scatter(x, data ["Log"] - linear (x, fitting_result.a), s=1)
	plt.xlabel('x[cm]')
	plt.ylabel(name + ' - y[i] - f(x[i]) [au]')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\" + "LinearResiduals.png")

	plt.clf()
	plt.plot(np.arange(max_D) + 1, test_err, 'b', label = 'test error')
	plt.plot(np.arange(max_D) + 1, train_err, 'r', label = 'train error')
	plt.legend()
	plt.xlabel("Degree")
	plt.ylabel("MSE")
	plt.title(name + "MSE(D)")
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\" + "MSE_D.png")
	#plt.show()

	Od1 = -1
	while (Od1 == -1):
		ROOT = tk.Tk()

		ROOT.withdraw()
		# the input dialog
		#Od = simpledialog.askstring(title="Polynom Order",
        #                          prompt="What's the polynom order ?")
		Od = 6
		# check it out
		#print("Hello", Od)
		try:
			Od1 = abs (int (Od))
		except ValueError:
			Od1 = -1
	best_poly = fit(fitting_data, fitting_functions_list.polynomial(int (Od1)))  # Do the actual fitting
	for i, ai in enumerate (best_poly.a):
		#print ("a[" + str (i) + "] = " + str (ai) + " \u00B1 " + str (best_poly.aerr [i]) + " (" + str (best_poly.arerr [i]) + "%)\n")
		table.write ("a[" + str (i) + "] = " + str (ai) + " \u00B1 " + str (best_poly.aerr [i]) + " (" + str (best_poly.arerr [i]) + "%)")
		if (i < len (best_poly.a) - 1):
			table.write ("\n")
	table.write ("\"," + str (best_poly.chi2_reduced) + "\n")
	print(best_poly)  # Print the results
	FitFile.write ("Best polynom is :"+ str (Od1) + "\n\n" + str (best_poly))
	FitFile.close ()
	plt.clf()
	#plt.errorbar(x, data ["Log"], yerr=data ["LogErr"], xerr=data ["xErr"], color="b")
	plt.scatter(x, data ["Log"], s=1)
	plt.plot(x, HighPoly (best_poly.a, Od1, x), color = "orange")
	plt.legend(["log(Av)", "6th Order Poly"], loc="upper right")
	plt.xlabel('x[cm]')
	plt.ylabel(nameNoExt + ' - Log (I) [au]')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\ " + str (Od1) + " BestPoly.png", dpi=1000)

	plt.clf()
	#plt.errorbar(x, data ["Log"] - HighPoly (best_poly.a, Od1, x), yerr=data ["LogErr"], xerr=data ["xErr"], color="b", capsize=3)
	plt.scatter(x, data ["Log"] - HighPoly (best_poly.a, Od1, x), s=1)
	#plt.plot (x, data ["Log"] - HighPoly (best_poly.a, Od1, x))
	plt.xlabel('x[cm]')
	plt.ylabel(nameNoExt + ' - y[i] - f(x[i]) [au]')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.savefig(sys.argv [1] + "\\" + nameNoExt + "\\ " + str (Od1) + " BestPolyResiduals.png", dpi=1000)

table.close ()

	

	
