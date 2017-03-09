import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

m_z = 91.19
s_w = (0.48076) #comphep's value for weinberg angle
c_w = np.sqrt(1-.48076**2)
c_v = -.5 + 2*s_w**2
c_a = -.5
alpha = 1/127.9 #0.0072973525664
factor = .3894 #millibarn
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
def qed_xsec(s):
    sigma = 4*np.pi*alpha**2/(4*s)
    qed_xsec_ = sigma*factor*1e9
    plt.figure(1)
    plt.plot(np.sqrt(s),qed_xsec_)
    plt.title(r'QED cross section for $e^+e^-\to\mu^+\mu^-$')
    plt.yscale('log')
    plt.xlabel(r'$\sqrt{s}$ [GeV]',size=15)
    plt.ylabel(r'$\sigma$',size=15)
    plt.savefig('qed_xsec.pdf')

def xsec(s):
    # Read CompHEP data
    infile = 'comphep_results/tab_3.txt'
    data = pd.read_csv(infile, sep=' ', skiprows=3, header=None)
    x = data[0]
    y = data[1]
    sigma = (4*np.pi*alpha**2/(3*s))*(1 + s**2/(16*(s_w*c_w)**4 * (s - m_z**2)**2) * (c_v**2 + c_a**2)**2 + s*c_v**2/(2*(s_w*c_w)**2*(s-m_z**2)) )
    xsec_ = sigma*factor*1e9
    plt.figure(2)
    plt.plot(np.sqrt(s),xsec_,x,y)
    plt.title(r'Electroweak cross section for $e^+e^-\to\mu^+\mu^-$')
    plt.yscale('log')
    plt.xlabel(r'$\sqrt{s}$ [GeV]',size=15)
    plt.ylabel(r'$\sigma_{QED}$',size=15)
    plt.legend(['Analytical','CompHEP'],loc=3)
    plt.savefig('xsec.pdf')
    return 0

def diff_xsec(s, ct, energy):
    # Read CompHEP data
    infile = None
    if energy == '5GeV':
        infile = 'comphep_results/tab_2.txt'
    elif energy == '125GeV':
        infile = 'comphep_results/tab_1.txt'
    else:
        infile = None
    data = pd.read_csv(infile, sep=' ', skiprows=3, header=None)
    x = data[0]
    y = data[1]
    dsigma = 2*np.pi*alpha**2/(4*s) * ( 1 + ct**2 + s**2/(16*(s-m_z**2)**2*(c_w*s_w)**4)*((1+ct**2)*(c_v**2 + c_a**2)**2 + 8*c_v**2*c_a**2*ct) + s/(2*(s-m_z**2)*(s_w*c_w)**2)*(c_v**2*(1+ct**2) + 2*c_a**2*ct) )
    diff_xsec_ = dsigma*factor*1e9
    plt.figure(3)
    plt.plot(ct,diff_xsec_,x,y)
    plt.title(r'Electroweak differential cross section for $e^+e^-\to\mu^+\mu^-$ at ' + energy)
    plt.yscale('log')
    plt.xlabel(r'$\cos\theta$',size=15)
    plt.ylabel(r'$d\sigma/d\cos\theta$',size=15)
    plt.legend(['Analytical','CompHEP'],loc=3)
    plt.savefig('diff_xsec.pdf')
    return 0

def asymmetry(s):
    # Read CompHEP data
    infile = 'comphep_results/tab_4.txt'
    data = pd.read_csv(infile, sep=' ', skiprows=3, header=None)
    x = data[0]
    y = data[1]
    sigma_F = 2*np.pi*alpha**2/(3*s) * (1 + (s**2 * ((c_v**2+c_a**2)**2 + 1/3.*c_v**2*c_a**2))/(16*(s_w*c_w)**4 * (s - m_z**2)**2) + s*(c_v**2 + 3/4.*c_a**2)/(2*(s_w*c_w)**2*(s-m_z**2)))
    sigma_B = 2*np.pi*alpha**2/(3*s) * (1 + (s**2 * ((c_v**2+c_a**2)**2 - 1/3.*c_v**2*c_a**2))/(16*(s_w*c_w)**4 * (s - m_z**2)**2) + s*(c_v**2 - 3/4.*c_a**2)/(2*(s_w*c_w)**2*(s-m_z**2)))
    sigma_F = sigma_F*factor*1e9
    sigma_B = sigma_B*factor*1e9
    asymmetry_ =  (sigma_F - sigma_B)/(sigma_F + sigma_B)
    plt.figure(4)
    plt.plot(np.sqrt(s),asymmetry_,x,y)
    plt.title(r'Asymmetry for cross section as function of $\sqrt{s}$')
    plt.xlabel(r'$\sqrt{s}$ [GeV]',size=15)
    plt.ylabel(r'$(\sigma_F - \sigma_B)/(\sigma_F + \sigma_B)$',size=15)
    plt.legend(['Analytical','CompHEP'],loc=3)
    plt.savefig('asym.pdf')

def angular_asymmetry(s,energy):
    theta_F     = np.linspace(0,np.pi/2,200)
    theta_B     = np.linspace(np.pi, np.pi/2,200)
    ct          = np.cos(theta_F)
    ct_B        = np.cos(theta_B)
    dcos_F  = np.pi*alpha**2/(4*s) * ( 1 + ct**2 + s**2/(16*(s_w*c_w)**4*(s - m_z**2)**2) * ((1+ct**2)*(c_v**2 + c_a**2)**2 + 8*c_v**2*c_a**2*ct) + s*(c_v**2*(1+ct**2) + 2*c_a**2*ct)/(2*(s_w*c_w)**2*(s-m_z**2)) )
    dcos_B  = np.pi*alpha**2/(4*s) * ( 1 + ct_B**2 + s**2/(16*(s_w*c_w)**4*(s - m_z**2)**2) * ((1+ct_B**2)*(c_v**2 + c_a**2)**2 + 8*c_v**2*c_a**2*ct) + s*(c_v**2*(1+ct_B**2) + 2*c_a**2*ct_B)/(2*(s_w*c_w)**2*(s-m_z**2)) )
    # dcos_F      = np.pi*alpha**2/(2*s) * (-ct - 1/3.*ct**3 + s**2/(16*(s_w*c_w)**4*(s - m_z**2)**2) * ((-ct - 1/3.*ct**3)*(c_v**2 + c_a**2)**2 - 4*c_v**2*c_a**2*ct**2) +  s/(2*(s_w*c_w)**2*(s-m_z**2)) * (c_v**2*(-ct - 1/3.*ct**3) - c_a**2*ct**2) )
    # dcos_B      = np.pi*alpha**2/(2*s) * (-ct_B - 1/3.*ct_B**3 + s**2/(16*(s_w*c_w)**4*(s - m_z**2)**2) * ((-ct - 1/3.*ct_B**3)*(c_v**2 + c_a**2)**2 - 4*c_v**2*c_a**2*ct_B**2) +  s/(2*(s_w*c_w)**2*(s-m_z**2)) * (c_v**2*(-ct_B - 1/3.*ct_B**3) - c_a**2*ct_B**2) )
    ang_asym = (dcos_F - dcos_B)/(dcos_F + dcos_B)
    plt.figure(5)
    plt.plot(ct, ang_asym)
    plt.title(r'Angular asymmetry for differential cross section at $\sqrt{s}=$'+energy)
    plt.xlabel(r'$\cos\,\theta$',size=15)
    plt.ylabel(r'$(d\sigma(\cos\theta_F) - d\sigma(\cos\theta_B))/(d\sigma(\cos\theta_F) + d\sigma(\cos\theta_B))$',size=15)
    plt.savefig('ang_asym.pdf')


s = np.linspace(5**2,200**2,100)
# Run functions to generate plots
qed_xsec(s)
xsec(s)
asymmetry(s)

s       = 5**2
theta_F = np.linspace(0, np.pi/2,200)
ct      = np.cos(theta_F)
# Choose energy to plot for:
# tab_1.txt: 125 GeV diff-xsec
# tab_2.txt:   5 GeV diff-xsec 
energy = '5GeV'
diff_xsec(s,ct,energy)
angular_asymmetry(s, energy)

s = np.array([5, 50, 91, 125])**2
sigma = (4*np.pi*alpha**2/(3*s))*(1 + s**2/(16*(s_w*c_w)**4 * (s - m_z**2)**2) * (c_v**2 + c_a**2)**2 + s*c_v**2/(2*(s_w*c_w)**2*(s-m_z**2)) )
xsec2 = sigma*factor*1e9
print("Cross section at some chosen energies:")
print("Energy: {}".format(np.sqrt(s)))
print("Xsec:   {}".format(xsec2))
print("All plots are saved as PDF files! ")