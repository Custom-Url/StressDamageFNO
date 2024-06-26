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
from neuralop.utils import UnitGaussianNormalizer



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


#########################
#########################


#################
#################
#################
def load_fracture_mesh_macrocurve( lst_dir0, dir_mesh,
                 n_train, n_tests,
                 batch_size, test_batch_sizes,
                 test_resolutions=[251],
                 train_resolution=251,
                 grid_boundaries=[[0,1],[0,1]],
                 positional_encoding=True,
                 encode_input=True,
                 encode_output=True,
                 encoding='channel-wise',
                 seq_len=1000):
    
    ## loop over directories - different vf, different UC, vp
    keyword = 'L0.05_vf'
    x = list()
    y = list()
    for dir0 in lst_dir0:  #different vf
        VF = re.search(f"{keyword}.*?(\d+\.\d+)", dir0).group(1)
        
        for dir1 in os.listdir(dir0):  #different UC, vp
            
            # initial geometry - mesh
            iuc = re.findall(r"\d+", dir1)[0]
            img_name = dir_mesh + '/' + keyword + VF + '/' + 'h0.0002' + '/' + dir1 + '.vtk'
            x.append(vtkFieldReader(img_name, 'tomo_Volume'))
            
            # stress-strain curves
            file_macro = datasets.utils.list_files(dir0 + '/' + dir1, ".std")[0]
            data_macro = read_macroStressStrain(dir0 + '/' + dir1 + '/' + file_macro)
            
            # unify the length
            eps = np.linspace(0, data_macro[-1,7], seq_len)
            sig = np.interp(eps, data_macro[:,7], data_macro[:,1])
            
            y.append(np.array([eps, sig]))

    ##
    x = np.expand_dims(np.array(x), 1)                       #BxCxWxHxD
    y = np.expand_dims(np.expand_dims(np.array(y),-1),-1)    #BxCxWxHxD, TODO: check if y needs normalisation
    
    idx_train = np.random.choice(len(x), n_train)  #random shuffle
    x_train = torch.from_numpy(x[idx_train,:,:,:,0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train,:,:,:,0]).type(torch.float32).clone()
    
    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]
    
    n_test = n_tests[0]  #currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)),idx_train), n_test)
    x_test = torch.from_numpy(x[idx_test,:,:,:,0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test,:,:,:,0]).type(torch.float32).clone()

    ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]
            
        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    ## output encoding
    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    ## training dataset 
    train_db = TensorDataset(x_train, y_train, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    ## test dataset
    test_db = TensorDataset(x_test, y_test,transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)
    
    test_loaders =  {train_resolution: test_loader}
    test_loaders[test_resolution] = test_loader #currently, only 1 resolution is possible

    return train_loader, test_loaders, output_encoder
    
    
    
    ############
    
    
    
    
def load_fracture_mesh_stress(lst_dir0, dir_mesh, n_train, n_tests, batch_size, test_batch_sizes,
                              test_resolutions=[251], train_resolution=251, grid_boundaries=[[0, 1], [0, 1]],
                              positional_encoding=True, encode_input=True, encode_output=True,
                              encoding='channel-wise', seq_len=1000):
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
            x.append(vtkFieldReader(img_name, 'tomo_Volume'))
            
            # output: stress (filename)
            registerFileName(lst_stress = out_field,
                             fprefix = dir0 + '/' + dir1 + '/' + LOADprefix,
                             loadstep = loadstep)

    # output: stress (data)
    nsamples = len(x)
    for i in range(nsamples):
        output = np.zeros((251, 251, 0))
        for key in out_field:
            output = np.concatenate( (output, vtkFieldReader(out_field[key][i],
                                                             fieldName=vtk_field_name(key))), axis=2 )
        y.append(output)
    
    ##
    x = np.expand_dims(np.array(x), 1)                       #BxCxWxHxD
    y = np.expand_dims(np.moveaxis(np.array(y),-1,1),-1)     #BxCxWxHxD
    
    idx_train = np.random.choice(len(x), n_train)  #random shuffle
    x_train = torch.from_numpy(x[idx_train,:,:,:,0]).type(torch.float32).clone()
    y_train = torch.from_numpy(y[idx_train,:,:,:,0]).type(torch.float32).clone()
    
    #
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]
    
    n_test = n_tests[0]  #currently, only 1 resolution possible
    idx_test = np.random.choice(np.setdiff1d(np.arange(len(x)),idx_train), n_test)
    x_test = torch.from_numpy(x[idx_test,:,:,:,0]).type(torch.float32).clone()
    y_test = torch.from_numpy(y[idx_test,:,:,:,0]).type(torch.float32).clone()

    ## input encoding
    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]
            
        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    ## output encoding
    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    ## training dataset 
    train_db = TensorDataset(x_train, y_train, transform_x=PositionalEmbedding2D(grid_boundaries) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    ## test dataset
    test_db = TensorDataset(x_test, y_test,transform_x=PositionalEmbedding2D(grid_boundaries) if positional_encoding else None)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)
    
    test_loaders =  {train_resolution: test_loader}
    test_loaders[test_resolution] = test_loader #currently, only 1 resolution is possible

    # Print statements to check the loaded data   
    print("Data loading completed successfully!")
    
    return train_loader, test_loaders, output_encoder

    
