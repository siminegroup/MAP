# MAP
An implementation of the Morphological Autoregressive Protocol (MAP) for generation of disordered molecular structures in 3D.

https://doi.org/10.1063/5.0174615

All required packages are listed in the requirements.txt file. The dataset is processed read in utils.py, followed by transform_data function, which voxelizes the input using an appropriate grid-point size. Based on our experience, setting the voxel size to approximately one-fourth of the typical bond length provides an optimal balance between spatial resolution and computational efficiency.


you can find the 2D implementations here:

https://github.com/InfluenceFunctional/GatedPixelCNN_v1

https://github.com/InfluenceFunctional/weld_net
