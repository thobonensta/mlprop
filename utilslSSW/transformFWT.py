import pywt
import numpy as np
from utilsSource.ComplexSourcePoint import CSP

def compressedFWT(u,family,level,Vs):
    ''' function that performs the FWT and compression with hard-threshold Vs'''
    U = pywt.wavedec(u,family,'per',level)
    U,slice = pywt.coeffs_to_array(U)
    U = pywt.threshold(U,Vs,mode='hard')
    U = pywt.array_to_coeffs(U,slice,output_format='wavedec')
    return U

if __name__ == '__main__':
    import matplotlib

    for backend in ['Qt5Agg', 'TkAgg', 'MacOSX', 'Agg']:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            print(f"Using backend: {matplotlib.get_backend()}")
            break
        except Exception as e:
            print(f"Failed to use backend {backend}: {e}")

    # Parameters (source)
    xs = -100
    zs = 50
    w0 = 5
    # Parameters (Propagation)
    x = 0
    f = 300e6
    c = 3e8
    wavelength = c / f
    k0 = 2 * np.pi * f / c
    zmax = 256
    # Discretization
    dz = 0.5 * wavelength
    z = np.arange(0, zmax, dz)
    Nz = int(zmax / dz)
    # Initial field
    u, _ = CSP(xs, zs, w0, x, k0, dz, Nz)
    # Compressed FWT and then IFWT
    U = compressedFWT(u,'sym6',3,1e-5)
    uw = pywt.waverec(U,'sym6',mode='per')
    plt.figure()
    plt.plot(20 * np.log10(np.abs(u) + 1e-15), z,label='u')
    plt.plot(20 * np.log10(np.abs(uw) + 1e-15), z,'--',label='uw')
    plt.grid()
    plt.legend()
    vmax = np.max(20 * np.log10(np.abs(u) + 1e-15)) + 1
    vmin = vmax - 70
    plt.xlim([vmin, vmax])
    plt.show()