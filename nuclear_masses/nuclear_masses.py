import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.mkdir(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.mkdir(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


infile = open(data_path("MassEval2016.dat"), 'r')


#####################################################################
#Data reading time!

Masses = pd.read_fwf(infile, usecols=(2, 3, 4, 6, 11),
            names=('N', 'Z', 'A', 'Element', 'Ebinding'),
            widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11, 9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 1),
            header=35,
            index_col=False)


Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce') #This line changes all the non numeric values to Nan
Masses = Masses.dropna() #Removes all rows with NaN
Masses['Ebinding'] /= 1000 #Just change from keV to MeV

Masses = Masses.groupby('A')
Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])

A = Masses['A']
Z = Masses['Z']
N = Masses['N']
Element = Masses['Element']
Energies = Masses['Ebinding']

#Setting up design matrix
X = np.zeros([len(A), 5])
X[:,0] = 1
X[:,1] = A
X[:,2] = A**(2.0/3.0)
X[:,3] = A**(-1.0/3.0)
X[:,4] = A**(-1.0)

clf = skl.LinearRegression().fit(X, Energies)
fity = clf.predict(X)
Masses['Eapprox'] = fity

fig, ax = plt.subplots()
ax.set_xlabel(r'$A = N + Z$')
ax.set_ylabel(r'$E_\mathrm{bind}\, /\mathrm{MeV}$')
ax.plot(Masses['A'], Masses['Ebinding'], alpha = 0.7, lw=2, label='Ame2016')
ax.plot(Masses['A'], Masses['Eapprox'], alpha = 0.7, lw=2, c='m', label='fit')

ax.legend()
ax.grid(1)
plt.show()
