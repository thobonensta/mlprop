from tqdm import tqdm
import numpy as np
from utilsSource.ComplexSourcePoint import CSP
from utilsSpace.phaseScreen import phasescreenwide
from utilsSpace.apodisation import HanningWindowUp
from utilslSSW.lSSW import lSSW
def get_triangle(d, h):
    ''' Function to model a triangular obstacle (staircase) '''
    x = np.linspace(0, d, d)
    y1 = [xi * h / (len(x) // 2) for xi in x[:len(x) // 2]]
    y2 = [xi * h / (-len(x) // 2) + 2 * h for xi in x[len(x) // 2:]]
    y = y1 + y2
    return y


def get_rectangle(d, h):
    ''' Function to model a rectangular obstacle (staircase) '''
    y = [h for _ in range(d)]
    return y

#
def get_element(d, h, el):
    ''' Function to create a given random obstacle '''
    if el == 0:
        return get_triangle(d, h)
    else:
        return get_rectangle(d, h)


def write_terrain_file(terrain0, ob,xmax,dx,Nx):
    ''' Function to create the given terrain for the number of obstacles chosen '''
    if ob == 1:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        el0 = get_element(int(terrain0[1]), int(terrain0[2]), int(round(terrain0[3], 0)))

        y = gap0 + el0

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1

    elif ob == 2:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        gap1 = [0 for _ in range(int(round(terrain0[1])))]
        el0 = get_element(int(terrain0[2]), int(terrain0[4]), int(round(terrain0[6], 0)))
        el1 = get_element(int(terrain0[3]), int(terrain0[5]), int(round(terrain0[7], 0)))

        y = gap0 + el0 + gap1 + el1

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1

    elif ob == 3:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        gap1 = [0 for _ in range(int(round(terrain0[1])))]
        gap2 = [0 for _ in range(int(round(terrain0[2])))]
        el0 = get_element(int(terrain0[3]), int(terrain0[6]), int(round(terrain0[9], 0)))
        el1 = get_element(int(terrain0[4]), int(terrain0[7]), int(round(terrain0[10], 0)))
        el2 = get_element(int(terrain0[5]), int(terrain0[8]), int(round(terrain0[11], 0)))

        y = gap0 + el0 + gap1 + el1 + gap2 + el2

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1

    elif ob == 4:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        gap1 = [0 for _ in range(int(round(terrain0[1])))]
        gap2 = [0 for _ in range(int(round(terrain0[2])))]
        gap3 = [0 for _ in range(int(round(terrain0[3])))]
        el0 = get_element(int(terrain0[4]), int(terrain0[8]), int(round(terrain0[12], 0)))
        el1 = get_element(int(terrain0[5]), int(terrain0[9]), int(round(terrain0[13], 0)))
        el2 = get_element(int(terrain0[6]), int(terrain0[10]), int(round(terrain0[14], 0)))
        el3 = get_element(int(terrain0[7]), int(terrain0[11]), int(round(terrain0[15], 0)))

        y = gap0 + el0 + gap1 + el1 + gap2 + el2 + gap3 + el3

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1

    elif ob == 5:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        gap1 = [0 for _ in range(int(round(terrain0[1])))]
        gap2 = [0 for _ in range(int(round(terrain0[2])))]
        gap3 = [0 for _ in range(int(round(terrain0[3])))]
        gap4 = [0 for _ in range(int(round(terrain0[4])))]

        el0 = get_element(int(terrain0[5]), int(terrain0[10]), int(round(terrain0[15], 0)))
        el1 = get_element(int(terrain0[6]), int(terrain0[11]), int(round(terrain0[16], 0)))
        el2 = get_element(int(terrain0[7]), int(terrain0[12]), int(round(terrain0[17], 0)))
        el3 = get_element(int(terrain0[8]), int(terrain0[13]), int(round(terrain0[18], 0)))
        el4 = get_element(int(terrain0[9]), int(terrain0[14]), int(round(terrain0[19], 0)))

        y = gap0 + el0 + gap1 + el1 + gap2 + el2 + gap3 + el3 + gap4 + el4

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1

    elif ob == 6:
        gap0 = [0 for _ in range(int(round(terrain0[0])))]
        gap1 = [0 for _ in range(int(round(terrain0[1])))]
        gap2 = [0 for _ in range(int(round(terrain0[2])))]
        gap3 = [0 for _ in range(int(round(terrain0[3])))]
        gap4 = [0 for _ in range(int(round(terrain0[4])))]
        gap5 = [0 for _ in range(int(round(terrain0[5])))]

        el0 = get_element(int(terrain0[6]), int(terrain0[12]), int(round(terrain0[18], 0)))
        el1 = get_element(int(terrain0[7]), int(terrain0[13]), int(round(terrain0[19], 0)))
        el2 = get_element(int(terrain0[8]), int(terrain0[14]), int(round(terrain0[20], 0)))
        el3 = get_element(int(terrain0[9]), int(terrain0[15]), int(round(terrain0[21], 0)))
        el4 = get_element(int(terrain0[10]), int(terrain0[16]), int(round(terrain0[22], 0)))
        el5 = get_element(int(terrain0[11]), int(terrain0[17]), int(round(terrain0[23], 0)))

        y = gap0 + el0 + gap1 + el1 + gap2 + el2 + gap3 + el3 + gap4 + el4 + gap5 + el5

        gap_1 = [0 for _ in range(int(xmax) - len(y))]
        y = y + gap_1


    ## generate elements to wrtie the input file
    ytxt = []
    for idx, yi in enumerate(y[1:]):
        if idx % dx == 0:
            ytxt.append(int(yi))
    ytxt = ytxt + [0 for _ in range(Nx - len(ytxt))]

    return ytxt

if __name__=='__main__':
    #---------------------------Initialization----------------------------------------#
    #-------------- Parameters -------------------------------------------------------#
    c = 3e8
    # Simulation parameters
    f = 300e6  # Simulation frequency [Hz]
    wavelength = c / f
    k0 = 2 * np.pi / wavelength
    xmax = 80000  # Maximal range for the simulation [m]
    zmax = 512  # Maximal altitude for the simulation [m]
    # Source parameters
    xs = - 50  # position of the source in x [m]
    zs = 70  # position of the source in z from the ground [m]
    w0 = 5 * wavelength  # width of the CSP
    # Discretization
    dx = 100 * wavelength
    dz = 0.5 * wavelength
    Nx = int(xmax / dx)
    Nz = int(zmax / dz)
    c = 0.1
    NimSSW = int(c * Nz)
    Napo = Nz
    # Source field (CSP placed at [xs,zs] with a width w0)
    x0 = 0
    u0, _ = CSP(xs, zs, w0, x0, k0, dz, Nz + Napo)
    # Polarisation
    polar = 'TE'
    # Ground parameters
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 1 / (c ** 2 * mu0)
    condG = 'Dielectric'  # condition of the ground [PEC or Dielectric or None]
    epsr1 = 1.0
    eps2 = 20
    sig2 = 0.02
    epsr2 = eps2 - 1j * sig2 / (2 * np.pi * f * epsilon0)
    #-------------- Compute the variation of the refraction index n with (x,z) -------#
    n = np.ones(Nz)
    ntot = np.zeros(Nz+Napo)
    ntot[:Nz] = n
    ntot[Nz:] = n[-1]

    # Defining the space operator
    L = phasescreenwide(dx,ntot,k0)
    # Defining the apodisation layer
    A = HanningWindowUp(zmax,dz,Nz,Napo)
    #
    Nobmin = 2
    Nobmax = 5
    for iob in range(Nobmin,Nobmax+1):

        terrains = np.loadtxt('./dataTerrain/TerrainGene{0}'.format(iob))
        data = []

        for idx, t in tqdm(enumerate(terrains)):
            # #----------------------------Model the terrain (relief)---------------------------#
            # # generate terrain profile
            y_true = write_terrain_file(t, iob,xmax,dx,Nx)
            y_true = np.array(y_true).astype('int')
            # # Propagation over the terrain
            family = 'sym6'
            level = 2
            Vs = 1e-3 * np.max(np.abs(u0))
            Vp = 1e-4
            remaining = NimSSW % (2 ** level)
            if remaining:  # if not zero
                NimSSW += (2 ** level) - remaining
            ussw = lSSW(u0, x0, zs, k0, epsr1, epsr2, dx, Nx, dz, Nz, NimSSW, Napo, L, A, y_true, polar, condG, family,
                        level, Vs, Vp)
            #------------------------------------------Target Data------------------------------#
            target = 20 * np.log10(abs(ussw[:,int(zs/dz)]) + 1e-15)
            target = target[:]
            y_true = y_true[:]
            data.append([y_true, target])
        ######################### SAVE THE DATA ######################################################
        data = np.array(data)
        arr0 = np.array([np.array(xi) for xi in data[:, 0]])
        arr1 = np.array([np.array(xi) for xi in data[:, 1]])
        st = np.zeros((arr0.shape[0], 2, Nx))
        st[:, 0, :] = arr0
        st[:, 1, :] = arr1
        np.savez('./dataField/FieldLabelGene{0}'.format(iob), data=st)
