# MOPAM
Repository for the IA (U-Net) based tropospheric propagation solver. 

For now the input are terrain (1d vectors) and the output is the field amplitude in dB at the altitude of the antenna (1d vectors). The network 
is trained in a supervised manner. 

The idea is to follow the physic model. Indeed, in the PWE solve with SSF/SSW
the terrain is modeled as a series of rectangles (staircase). Thus, we choose to 
train the model using only two kind of obstacles, either rectangles or triangles.
The first one convey the staircase structure while the second allows to have some
obstacles with diffraction. To uniformly sample the underlying distribution we
construct ramdom terrain consisting of 2 to 5 obstacles of different sizes and widths.
We also use different gap between the obstacles. This is performed first using a
Latin Hypercube Sampling Strategy and now using its orthonormal variation. We also choose
to balance the different terrains, i.e. as many terrains with 2 obstacles as with 5.
It is not perfect since the edge meaning obstacles under the antenna are not accounted for. This limit 
for now the method to antenna over a flat ground with no direct descending relief, and 
is a kind of normalization. Then the training is performed over a set of labeled field (dB) with
the associated terrain at the altitude of the antenna. We use a combine L2 and L1 loss. We did not
add noise to translate all terrain and convey the fact that the field is translation invariant
in this case, which could help generalization. In the papers, we also introduce a fine tuning to
adapt the network at a low cost compared to a full training. In addition some work on conformal
prediction have been done, but not included yet in this repository.

# Git command
To clone the repository in order to have a branch to date and be able to modify the command in terminal is
> git clone https://github.com/thobonensta/propaSSFandSSW.git

This will create the associated repository in the file you are. Then you can modify everything and when you
want you can add, commit and push. If you create a new file, you have to
> git add filename

If you modify or add or remove you will then have to
> git commit -m "what did you do"

After that you can push the modifications
> git push 

If you are using PyCharmPro it can be done directlt through the interface of PyCharm.

**Nonetheless, to not add or push the DATA since they will be too big for git push directly !!!**

# Python (needed libraries)
The code has been tested on Mac and Windows for python3.10 with scipy 1.13.0 (for the LHS sampling)

# Documentation

mlprop is comprised of 8 repository and one main file.
<ul>
<li> <strong>utilsSSW</strong> which contains all the necessary functions to perform the local SSW method</li>
<li> <strong>utilsRefraction</strong> which contains the code for the linear and trilinear refractive profile</li>
<li> <strong>utilsRelief</strong> that contains the functions to perform the staircaze model and then the translations</li>
<li> <strong>utilsSource</strong> which contains the file to compute the initial field</li>
<li> <strong>utilsSpace</strong> that contains the operators that are performed in the spatial domain</li>
<li> <strong>utilsSSF</strong> which contains all the necessary function for SSF (discrete wide angle but also continuous if wanted)</li>
<li> <strong>data</strong> which contains all the necessary function to compute the data and some data</li>
<li> <strong>aiModel</strong> which contains the network model (UNet), the training process and some data with a network trained</li>
</ul>	
main.py allows to test the code for a pre-trained network.

