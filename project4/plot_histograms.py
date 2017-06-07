#!/usr/bin/env python3

"""
Plotting data run through BDT made in BDT_refined.py

!!! Be aware of the different arrays that should be given
when using model with/without the transverse and
invariant masses. !!!

@author Jon Vegard Sparre
May 2017
"""

import numpy as np
import matplotlib.pyplot as plt
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(**params)
import pandas as pd
import sklearn as sk
from sklearn import ensemble
import textwrap, sys

print('Pandas version ' + pd.__version__)
print('Sklearn version ' + sk.__version__)

# Reading in data
egamma     = pd.read_csv('egamma_data_alljets_onebjet.txt', skipinitialspace=True)
muons      = pd.read_csv('muons_data_alljets_onebjet.txt', skipinitialspace=True)
zprime     = pd.read_csv('zprime_data_alljets_onebjet.txt', skipinitialspace=True)

# Create data frames in pandas
df_egamma = pd.DataFrame(egamma)
df_muons  = pd.DataFrame(muons)
df_zprime = pd.DataFrame(zprime)

# Fetching values for chosen variables
feature_list = ["lepton.pt()", "lepton.eta()", "lepton.phi()", "b_jet.pt()", "b_jet.eta()", "b_jet.phi()", "etmiss.et()", "etmiss.phi()", "TransMass", "InvMass"]
# Array with all values for plotting
egamma_features_plot = df_egamma[feature_list].values
zprime_features_plot = df_zprime[feature_list].values
muons_features_plot  = df_muons[feature_list].values

# Arrays without the tranverse and invariant masses for the models without these variables
# egamma_features = df_egamma[feature_list[0:8]].values
# zprime_features = df_zprime[feature_list[0:8]].values
# muons_features  = df_muons[feature_list[0:8]].values

# Use these if the model is trained with transverse and invariant masses
egamma_features = egamma_features_plot
zprime_features = zprime_features_plot
muons_features  = muons_features_plot

# Load model
from sklearn.externals import joblib
# clf = joblib.load('AllData_depth1_n_estimators100.pkl')            # Model without tranverse and invariant masses
clf = joblib.load('AllData_depth5_n_estimators100_TransInv.pkl') # Model with tranverse and invariant masses

# Predict occurences of top quark in egamma data set
pred_zprime = clf.predict(zprime_features)
pred_0_zprime = np.where(pred_zprime == 0) # Class 0: no top quark
pred_1_zprime = np.where(pred_zprime == 1) # Class 1: top quark

pred_egamma = clf.predict(egamma_features)
pred_0_egamma = np.where(pred_egamma[:] == 0)
pred_1_egamma = np.where(pred_egamma[:] == 1)

pred_muons = clf.predict(muons_features)
pred_0_muons = np.where(pred_muons[:] == 0)
pred_1_muons = np.where(pred_muons[:] == 1)

# Make styleful plots
plt.style.use('ggplot')

def plot_histogram(number, title, xlabel, ylabel, data, bins_, name, xlim, vertical=None, verticallabel=None):
    ''' Plotting histograms of data from the BDT.
    '''
    plt.figure(number)
    plt.hist(data, bins=bins_, range=xlim)
    if vertical:
        plt.axvline(vertical, color='black', linestyle='dashed', label=verticallabel)
    plt.title("\n".join(textwrap.wrap(title, 50)))
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(name)
    return 0


bins_ = 100
xlim_trans = [0,300]
xlim_inv = [0,300]
m_W = 80.385 # [GeV]
m_bl = np.sqrt(170**2 - 80.385**2)

# Invariant mass plots
plot_histogram(4, r"Invariant mass of lepton and b-jet, Z' data set", r'$m_{b\ell}$ [GeV]',
    r'No. of events without top', zprime_features_plot[pred_0_zprime[0],9][:], bins_, "zprime_hist_0_inv.pdf",
    xlim_inv, vertical=np.sqrt(170**2 - 80.385**2), verticallabel=r'$\sqrt{m_t^2 - m_W^2}$')

plot_histogram(5, r"Invariant mass of lepton and b-jet, Z' data set", r'$m_{b\ell}$ [GeV]',
    r'No. of events with top', zprime_features_plot[pred_1_zprime[0],9][:], bins_, "zprime_hist_1_inv.pdf",
    xlim_inv, vertical=np.sqrt(170**2 - 80.385**2), verticallabel=r'$\sqrt{m_t^2 - m_W^2}$')

# # Transverse mass plots
# plot_histogram(8, r"Transverse mass of lepton and missing energy, Z' data set", r'$m_{T}$ [GeV]',
#     r'No. of events without top', zprime_features[pred_0_zprime[0],8][:], bins_, "zprime_hist_0_trans.pdf", 
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(9, r"Transverse mass of lepton and missing energy, Z' data set", r'$m_{T}$ [GeV]',
#     r'No. of events with top', zprime_features[pred_1_zprime[0],8][:], bins_, "zprime_hist_1_trans.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(10, r"Transverse mass of lepton and missing energy, eGamma data set", r'$m_{T}$ [GeV]',
#     r'No. of events without top', egamma_features[pred_0_egamma[0],8][:], bins_, "egamma_hist_0_trans.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(11, r"Transverse mass of lepton and missing energy, eGamma data set", r'$m_{T}$ [GeV]',
#     r'No. of events with top', egamma_features[pred_1_egamma[0],8][:], bins_, "egamma_hist_1_trans.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(12, r"Transverse mass of lepton and missing energy, muons data set", r'$m_{T}$ [GeV]',
#     r'No. of events without top', muons_features[pred_0_muons[0],8][:], bins_, "muons_hist_0_trans.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(13, r"Transverse mass of lepton and missing energy, muons data set", r'$m_{T}$ [GeV]',
#     r'No. of events with top', muons_features[pred_1_muons[0],8][:], bins_, "muons_hist_1_trans.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

# plot_histogram(23, r"Transverse mass of lepton and missing energy, Z' data set", r'$m_{T}$ [GeV]',
#     r'No. of events with top', zprime_features[:,8][:], bins_, "lol.pdf",
#     xlim_trans, vertical=80.385, verticallabel=r'$m_W$')

plt.figure(1)
plt.hist(muons_features_plot[pred_1_muons[0],9], bins=bins_, range=xlim_trans, label=r'$\mu$ With $t$')
plt.hist(egamma_features_plot[pred_1_egamma[0],9], bins=bins_, range=xlim_trans, label=r'$e\gamma$ With $t$')
plt.hist(muons_features_plot[pred_0_muons[0],9], bins=bins_, range=xlim_trans, label=r'$\mu$ Without $t$')
plt.hist(egamma_features_plot[pred_0_egamma[0],9], bins=bins_, range=xlim_trans, label=r'$e\gamma$ Without $t$')
plt.axvline(m_bl, color='black', linestyle='dashed', label=r'$\sqrt{m_t^2 - m_W^2}$')
plt.title("\n".join(textwrap.wrap(r"Invariant mass of lepton and $b$-jet, eGamma and eMuons data set", 50)))
plt.xlim(xlim_trans)
plt.xlabel(r'$m_{b\ell}$ [GeV]')
plt.ylabel(r'No. of events')
plt.legend()
plt.savefig('egamma_emuons_invmass_01_depth1.pdf')