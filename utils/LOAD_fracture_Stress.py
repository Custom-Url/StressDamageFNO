"""
LOAD_Fracture_V2
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import re
import os

from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
from neuralop.datasets.output_encoder import UnitGaussianNormalizer

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

import cv2



# =====================
# some helper functions
# =====================
def vtkFieldReader(vtk_name, fieldName='tomo_Volume'):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_name)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    siz = list(dim)
    siz = [i - 1 for i in siz]
    mesh = vtk_to_numpy(data.GetCellData().GetArray(fieldName))
    return mesh.reshape(siz, order='F')


def read_macroStressStrain(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = list()
    for line in lines[6:]:
        data.append( [float(num) for num in line.split()] )
    return np.array(data)


def registerFileName(lst_stress=None, lst_strain=None, lst_damage=None, fprefix=None, loadstep=None, zeroVTK=False):
    if zeroVTK is False:
        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')
            
        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')
            
        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(fprefix + '_' + key + '_' + str(loadstep) + '.vtk')
    else:
        
        zeroVTKlocation = 'zerovtk.vtk'
        
        if lst_stress is not None:
            for key in lst_stress:
                lst_stress[key].append(zeroVTKlocation)
            
        if lst_strain is not None:
            for key in lst_strain:
                lst_strain[key].append(zeroVTKlocation)
        
        if lst_damage is not None:
            for key in lst_damage:
                lst_damage[key].append(zeroVTKlocation)


def init_dict_StressStrainDamage():
    dict_stress = {'sig1': list(),
                   'sig2': list(),
                   'sig4': list()}
    dict_strain = {'def1': list(),
                   'def2': list(),
                   'def4': list()}
    dict_damage = {'M1_varInt1': list(),
                   'M2_varInt1': list()}
    
    return dict_stress, dict_strain, dict_damage

def vtk_field_name(key):
    if key == 'sig1':
        return 'Sig_1'
    elif key == 'sig2':
        return 'Sig_2'
    elif key== 'sig4':
        return 'Sig_4'
    elif key == 'def1':
        return 'Def_1'
    elif key == 'def2':
        return 'Def_2'
    elif key == 'def4':
        return 'Def_4'
    elif key== 'M1_varInt1':
        return 'M1_varInt1'
    elif key == 'M2_varInt1':
        return 'M2_varInt1'
    else:
        assert "key unknown, sorry"
        
   
def load_fracture_mesh_stress(lst_dir0, dir_mesh, 
                              train_resolution,
                              n_train, n_tests, 
                              batch_size, test_batch_sizes,
                              test_resolutions, 
                              grid_boundaries=[[0, 1], [0, 1]],
                              positional_encoding=True, 
                              encode_input=True, 
                              encode_output=True,
                              encoding='channel-wise',
                              channel_dim=1,
                              num_workers=2,
                              pin_memory=True, 
                              persistent_workers=True,
                              target_resolution=(251, 251)
                              ):    
    

    
    print("Starting data loading...")
    
    # create empty lists for input / output
    out_field, _, _ = init_dict_StressStrainDamage()
    
    # loop over directories - different vf, different UC, vp
    keyword = 'L0.05_vf'
    LOADprefix = 'Load0.0'
    loadstep = 10
    x = list()
    y = list()
    for dir0 in lst_dir0:  #different vf
        VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)
        for dir1 in os.listdir(dir0):  #different UC, vp
            # input: mesh (data)
            iuc = re.findall(r"\d+", dir1)[0]
            img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'
            # Load the image
            image = vtkFieldReader(img_name, 'tomo_Volume')
            # Resize the image to target resolution
            resized_image = cv2.resize(image, target_resolution)
            resized_image = np.expand_dims(resized_image, axis=-1)
            x.append(resized_image)
           
                        
            # output: stress (filename)
            registerFileName(lst_stress = out_field,
                              fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                              loadstep = loadstep)
    
###############################################################################

    # output: stress (data)
    nsamples = len(x)
    for i in range(nsamples):
        output = np.zeros((target_resolution[0], target_resolution[1], 0))
        for key in out_field:      

            mesh =  vtkFieldReader(out_field[key][i], fieldName=vtk_field_name(key))
            mesh = np.expand_dims((cv2.resize(mesh, target_resolution)), axis=-1)
            output = np.concatenate( (output, mesh), axis=2 )           
        y.append(output)
    
    x = np.expand_dims(np.array(x), 1)                      
    y = np.expand_dims(np.moveaxis(np.array(y),-1,1),-1)     

    idx_train = np.random.choice(len(x), n_train)  #random shuffle
    x_train = torch.from_numpy(x[idx_train,:,:,:,0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train,:,:,:,0]).type(torch.float32).clone()
    
    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]
    
    n_test = n_tests[0]  #currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)),idx_train), n_test)
    ###############################
    idx_test = np.arange(0, len(x))
    ###############################
    x_test = torch.from_numpy(x[idx_test,:,:,:,0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test,:,:,:,0]).type(torch.float32).clone()
    
    pos_encoding = None

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
            print("x_train ndim", x_train.ndim)
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        input_encoder.transform(x_test)
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        output_encoder.transform(y_test)
    else:
        output_encoder = None

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)  
                            
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries)

    data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                          out_normalizer=output_encoder,
                                          positional_encoding=pos_encoding)
                
    ## training dataset 
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_db,
                                                batch_size=batch_size, shuffle=True, drop_last=True,
                                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_db,
                                               batch_size=test_batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    test_loaders =  {train_resolution: test_loader}
    test_loaders[test_resolution] = test_loader #currently, only 1 resolution is possible

    # Print statements to check the loaded data   
    print("Data loading completed successfully!")

    # # Plot some samples from the input data
    # num_samples_to_plot = 1 # Change this value as needed
    # fontsize = 10

    # fig = plt.figure(figsize=(20, 10*(num_samples_to_plot + 1)))

    # # Define a grid with 5 rows (2 for plots, 2 for color bars, and 1 for spacing) and 4 columns
    # gs = GridSpec(2, 4, height_ratios=[1, 0.1])

    # for i in range(num_samples_to_plot):
    #     # Move tensors to CPU and convert to NumPy arrays
    #     input_data = x_train[i].cpu().numpy()
    #     output_data = y_train[i].cpu().numpy()

    #     # Plot input data
    #     axs_input = fig.add_subplot(gs[i, 0])
    #     axs_input.imshow(input_data.squeeze())  
    #     axs_input.set_title(f'Input Sample', fontsize=fontsize)

    #     # Plot corresponding xx stress data
    #     axs_xx = fig.add_subplot(gs[i, 1])
    #     im_xx = axs_xx.imshow(output_data.squeeze()[0])  
    #     axs_xx.set_title(f'XX Stress Sample', fontsize=fontsize)

    #     # Plot corresponding yy stress data
    #     axs_yy = fig.add_subplot(gs[i, 2])
    #     im_yy = axs_yy.imshow(output_data.squeeze()[1])  
    #     axs_yy.set_title(f'YY Stress Sample', fontsize=fontsize) 

    #     # Plot corresponding xy stress data
    #     axs_xy = fig.add_subplot(gs[i, 3])
    #     im_xy = axs_xy.imshow(output_data.squeeze()[2])  
    #     axs_xy.set_title(f'XY Stress Sample', fontsize=fontsize) 

    #     # Define labels for color bars
    #     labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\sigma_{xy}$']

    # # Add colorbars
    # for col, label in zip(range(1, 4), labels):
    #     cbar_ax = fig.add_subplot(gs[num_samples_to_plot, col])
    #     cbar = plt.colorbar(im_xx if col == 1 else im_yy if col == 2 else im_xy, cax=cbar_ax, orientation='horizontal')
    #     cbar.set_label(label, fontsize=fontsize+5)

    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.5, top=0.9, wspace=0.5, hspace=0.2)
    # plt.show()
    
        
    return train_loader, test_loaders, data_processor

    
