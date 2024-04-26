"""
StressDamage_Load
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        
# Function to extract loadstep number from filename
def extract_loadstep(filename):
    # Split filename by underscores and dots to handle various formats
    sections = filename.split('_')
    for section in sections:
        # Check if section contains a dot
        if '.vtk' in section:
            parts = section.split('.')
            for part in parts:
                if part.isdigit():
                    return int(part)
    # If loadstep number is not found, return a default value or handle the error as needed
    return None

# Function to find the maximum loadstep for a given directory
def find_max_loadstep(directory):
    filenames = os.listdir(directory)
    loadsteps = [extract_loadstep(filename) for filename in filenames if extract_loadstep(filename) is not None]    
    return max(loadsteps)



#########################
#########################

   
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
                              normalization='unit_gaussian'
                              ):    
    
   
   
    print("Starting data loading...")
    
    # create empty lists for input / output
    stress_field, _, damage_field = init_dict_StressStrainDamage()
    out_field = {**stress_field, **damage_field}
    
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
            x.append(vtkFieldReader(img_name, 'tomo_Volume'))
            
            # Find the highest loadstep for the current directory
            max_loadstep = find_max_loadstep(os.path.join(dir0, dir1))
            
            # output: stress (filename)
            registerFileName(lst_stress = stress_field,
                              fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                              loadstep = loadstep)

            # output: damage (filename)
            registerFileName(lst_stress = damage_field,
                              fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                              loadstep = max_loadstep)

    nsamples = len(x)
    for i in range(nsamples):
        stress_output = np.zeros((251, 251, 0))
        damage_output = np.zeros((251, 251, 0))

        for key in stress_field:        
            stress_output = np.concatenate( (stress_output, vtkFieldReader(stress_field[key][i], fieldName=vtk_field_name(key))), axis=2 ) 

        for key in damage_field:    
            if key == 'M1_varInt1':
                fibre_damage = np.concatenate( (damage_output, vtkFieldReader(damage_field[key][i], fieldName=vtk_field_name(key))), axis=2 )           
            elif key == 'M2_varInt1':
                mesh_damage = np.concatenate( (damage_output, vtkFieldReader(damage_field[key][i], fieldName=vtk_field_name(key))), axis=2 )
        damage_output = fibre_damage + mesh_damage

        output = np.concatenate((stress_output, damage_output), axis=2)
        y.append(output)
      
           
    ##
    x = np.expand_dims(np.array(x), 1)                       #BxCxWxHxD
    y = np.expand_dims(np.moveaxis(np.array(y),-1,1),-1)     #BxCxWxHxD
    
    idx_train = np.random.choice(len(x), n_train)  #random shuffle
    x_train = torch.from_numpy(x[idx_train,:,:,:,0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train,:,:,:,0]).type(torch.float32).clone()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    
    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]
    
    n_test = n_tests[0]  #currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)),idx_train), n_test)
    x_test = torch.from_numpy(x[idx_test,:,:,:,0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test,:,:,:,0]).type(torch.float32).clone()
    
    pos_encoding = None


     ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]
        if normalization == 'min_max':
            # Apply min-max scaling first
            # print("Mean before min-max scaling (input):", x_train.mean().item())
            # print("Standard deviation before min-max scaling (input):", x_train.std().item())
            # x_train = min_max_scaling(x_train)
            # x_test = min_max_scaling(x_test)
            # # Apply standardization to make mean 0 and std 1
            # print("Mean after min-max scaling (input):", x_train.mean().item())
            # print("Standard deviation after min-max scaling (input):", x_train.std().item())
            # x_train = standardization(x_train)
            # x_test = standardization(x_test)
            # print("Mean after standardization (input):", x_train.mean().item())
            # print("Standard deviation after standardization (input):", x_train.std().item())
            input_encoder = None  # No input encoder needed for min-max normalization
        elif normalization == 'unit_gaussian':
            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            x_train = input_encoder.fit(x_train).transform(x_train).float()
            x_test = input_encoder.transform(x_test).float()

            max_value = x_train.max().item()  # Get the maximum value from the training data
            x_train /= max_value
            x_test /= max_value
        else:
            raise ValueError("Invalid normalization technique specified.")
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]
        if normalization == 'min_max':
            # Apply min-max scaling first
            # print("Mean before min-max scaling (output):", y_train.mean().item())
            # print("Standard deviation before min-max scaling (output):", y_train.std().item())
            # y_train = min_max_scaling(y_train)
            # y_test = min_max_scaling(y_test)
            # print("Mean after min-max scaling (output):", y_train.mean().item())
            # print("Standard deviation after min-max scaling (output):", y_train.std().item())
            # # Apply standardization to make mean 0 and std 1
            # y_train = standardization(y_train)
            # y_test = standardization(y_test)
            # print("Mean after standardization (output):", y_train.mean().item())
            # print("Standard deviation after standardization (output):", y_train.std().item())
            output_encoder = None
        elif normalization == 'unit_gaussian':
            print("Mean before normalization (output):", y_train.mean().item())
            print("Standard deviation before normalization (output):", y_train.std().item())
            # Create a UnitGaussianNormalizer instance
            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            y_train = output_encoder.fit(y_train).transform(y_train).float()
            y_test = output_encoder.transform(y_test).float()

            max_value = y_train.max().item()  # Get the maximum value from the training data
            y_train /= max_value
            y_test /= max_value

            # Print the data type of the normalized tensors
            print("DTYPE (y_train):", y_train.dtype)
            print("DTYPE (y_test):", y_test.dtype)
            print("Mean after normalization (output):", y_train.mean().item())
            print("Standard deviation after before normalization (output):", y_train.std().item())
        else:
            raise ValueError("Invalid normalization technique specified.")
    else:
        output_encoder = None

                           
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries)


    data_processor = DefaultDataProcessor(in_normalizer=None,
                                          out_normalizer=None,
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
    # num_samples_to_plot = 3  # Change this value as needed
    
    # fig, axs = plt.subplots(num_samples_to_plot, 5, figsize=(120, 80))
    
    # for i in range(num_samples_to_plot):
    #     # Move tensors to CPU and convert to NumPy arrays
    #     input_data = x_train[i].cpu().numpy()
    #     output_data = y_train[i].cpu().numpy()
    
    #     # Plot input data
    #     axs[i, 0].imshow(input_data.squeeze())  
    #     axs[i, 0].set_title(f'Input Sample {i+1}')
    
    #     # Plot corresponding xx stress data
    #     axs[i, 1].imshow(output_data.squeeze()[0])  
    #     axs[i, 1].set_title(f'XX Stress Sample {i+1}')

    #     # Plot corresponding yy stress data
    #     axs[i, 2].imshow(output_data.squeeze()[1])  
    #     axs[i, 2].set_title(f'YY Stress Sample {i+1}') 

    #     # Plot corresponding xy stress data
    #     axs[i, 3].imshow(output_data.squeeze()[2])  
    #     axs[i, 3].set_title(f'XY Stress Sample {i+1}') 

    #     # Plot corresponding damage data
    #     axs[i, 4].imshow(output_data.squeeze()[3])  
    #     axs[i, 4].set_title(f'Damage Sample {i+1}')  
    
    # plt.tight_layout()
    # plt.show()
    
    return train_loader, test_loaders, data_processor

    
