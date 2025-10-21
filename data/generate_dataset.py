from scipy.stats import qmc
import numpy as np
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt

def generateDataTerrain(dim,n,dmax,hmin,hmax):
    ''' function that generate the date for a certain number of obstacles
    for each obstacle we have 4 parameters (gap from previous one, width, altitude and which)
    it can be either a rectangle or a triangle for now
    n shall be a prime number sufficient to have a low discrepancy and represent well the
    underlying manifold
    '''
    # Number of obstacle
    Nob = dim//4
    # divide the size uniformly between the obstacles
    tmpSize = dmax/4
    # 1/4 of the size for the gap
    tmpgapmax = 1/4*tmpSize
    # 3/4 of the size for the width
    tmpWidthmax = 3/4*tmpSize
    # Calculates all the bound for the sampling
    gapBoundMin = [0 for _ in range(Nob)]
    WidthBoundMin = [100 for _ in range(Nob)]
    hBoundMin = [hmin for _ in range(Nob)]
    elBoundMin = [0 for _ in range(Nob)]
    gapBoundMax = [tmpgapmax for _ in range(Nob)]
    WidthBoundMax = [tmpWidthmax for _ in range(Nob)]
    hBoundMax = [hmax for _ in range(Nob)]
    elBoundMax = [1 for _ in range(Nob)]
    # Concatenate all the lower and upper bounds
    l_bounds = gapBoundMin+WidthBoundMin+hBoundMin+elBoundMin
    u_bounds = gapBoundMax+WidthBoundMax+hBoundMax+elBoundMax
    # Sample quasi uniformly the boundaries using qmc
    sampler = qmc.LatinHypercube(d=dim, strength=2, optimization="random-cd")
    sample = sampler.random(n)
    variables = qmc.scale(sample, l_bounds, u_bounds)
    # discrepancy to see if the random state is good
    disc = qmc.discrepancy(sample)
    # save in a text file the data
    np.savetxt('./dataTerrain/TerrainGene{0}'.format(Nob),variables)

    return np.array(variables),disc


# constants
dmax = 80000


## generate variables for a 2 obstacles problem
sampler = qmc.LatinHypercube(d=8,strength=2,optimization="random-cd")
sample = sampler.random(n=1369)
#       gap0, gap1 d1, d2, h1, h2, el1, el2
l_bounds = [0, 500, 1000, 1000, 10, 10, 0, 0]
u_bounds = [10000, 10000, 30000, 30000, 65, 65, 1, 1]
variables = qmc.scale(sample, l_bounds, u_bounds)
print(qmc.discrepancy(sample))
#np.savetxt('terrain_generation_2_obstacles_1369.txt',variables)


## generate variables for a 3 obstacles problem
sampler = qmc.LatinHypercube(d=12,strength=2,optimization="random-cd")
sample = sampler.random(n=1369)
# gap0, gap1, gap2 d1, d2, d3, h1, h2, h3, el1, el2, el3
l_bounds = [0, 500, 500, 1000, 1000, 1000, 10, 10, 10, 0, 0, 0]
u_bounds = [5000, 5000, 5000, 20000, 20000, 20000, 65, 65, 65, 1, 1, 1]
variables = qmc.scale(sample, l_bounds, u_bounds)
print(qmc.discrepancy(sample))
#np.savetxt('terrain_generation_3_obstacles_1369.txt',variables)

## generate variables for a 4 obstacles problem
sampler = qmc.LatinHypercube(d=16,strength=2,optimization="random-cd")
sample = sampler.random(n=1369)
# gap0, gap1, gap2, gap3 d1, d2, d3, d4, h1, h2, h3, h4, el1, el2, el3, el4
l_bounds = [0, 500, 500, 500, 1000, 1000, 1000, 1000, 10, 10, 10, 10, 0, 0, 0, 0]
u_bounds = [3000,3000,3000,3000,17000,17000,17000,17000,65, 65, 65, 65, 1, 1, 1,1]
variables = qmc.scale(sample, l_bounds, u_bounds)
print(qmc.discrepancy(sample))
#np.savetxt('terrain_generation_4_obstacles_1369.txt',variables)


## generate variables for a 5 obstacles problem
sampler = qmc.LatinHypercube(d=20,strength=2,optimization="random-cd")
sample = sampler.random(n=1369)
# gap0, gap1, gap2, gap3, gap4, d1, d2, d3, d4, d5, h1, h2, h3, h4, h5, el1, el2, el3, el4, el5
l_bounds = [0, 500, 500, 500, 500,1000,1000,1000,1000,1000 , 10, 10, 10, 10, 10, 0, 0, 0, 0,0]
u_bounds = [2000,2000,2000,2000,2000,14000,14000,14000,14000,14000,65, 65, 65, 65, 65,1, 1, 1,1,1]
variables = qmc.scale(sample, l_bounds, u_bounds)
print(qmc.discrepancy(sample))
#np.savetxt('terrain_generation_5_obstacles_1369.txt',variables)

# 6 obstacles
sampler = qmc.LatinHypercube(d=24,strength=2,optimization="random-cd")
sample = sampler.random(n=(23)**2)
# # gap0, gap1, gap2, gap3, gap4, gap5, d1, d2, d3, d4, d5, d6, h1, h2, h3, h4, h5, h6, el1, el2, el3, el4, el5, el6
l_bounds = [0, 500, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0]
u_bounds = [5000,5000,5000,5000,5000,5000,12000,12000,12000,12000,12000,12000,65, 65, 65, 65, 65, 65, 1, 1, 1, 1, 1, 1]
variables = qmc.scale(sample, l_bounds, u_bounds)
#np.savetxt('terrain_generation_6_obstacles_211.txt',variables)



var, disc = generateDataTerrain(8,1369,dmax,10,65)

# To see the qmc sampling
plt.figure()
plt.scatter(var[:,0],var[:,1])
plt.figure()
plt.scatter(var[:,-2],var[:,-1])
plt.show()

print(disc)




