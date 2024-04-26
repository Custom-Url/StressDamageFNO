"""
Can the Fourier Neural Operator Achieve Super Resolution for the Prediction of 
Stress and Damage Fields in Composites?
=======================================

Training a TFNO on Stress and Damage Fields in Composites
=========================================================

Dylan Gray
MEng (Hons) Integrated Mechanical and Electrical Engineering
University of Bath
2023/2024

"""
import sys
import os
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from neuralop import Trainer, get_model
from neuralop import LpLoss, H1Loss
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup, BasicLoggerCallback

from utils.LOAD_fracture_Damage import load_fracture_mesh_stress

from neuralop.utils import count_model_params

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

import wandb

from neuralop.models.fno import FNO2d

from sklearn.metrics import f1_score


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./Damage_TFNO_config.yaml", config_name="default", config_folder="../dg765/config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# # Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
    wandb.login(key="insert-wandb-key")
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                "Damage",
                config.data.batch_size,
                config.data.n_train,
                config.data.n_tests,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Cuda Available')

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# List of directories for stress/damage fields
lst_dir0 = list()
idx_UC = list()

shuffle = True
# PC FILEPATH
#-------------------------------------------------------
# lst_dir0.append( r"F:/FYP/data/L0.05_vf0.25/h0.0002" )
# lst_dir0.append( r"F:/FYP/data/L0.05_vf0.3/h0.0002" )
# lst_dir0.append( r"F:/FYP/data/L0.05_vf0.5/h0.0002" )
# lst_dir0.append( r"F:/FYP/data/L0.05_vf0.6/h0.0002" )
# dir_mesh = r"F:/FYP/data/mesh"

#LAPTOP FILEPATH
#-------------------------------------------------------
# lst_dir0.append( r"D:/FYP/data/L0.05_vf0.25/h0.0002" )
# lst_dir0.append( r"D:/FYP/data/L0.05_vf0.3/h0.0002" )
# lst_dir0.append( r"D:/FYP/data/L0.05_vf0.5/h0.0002" )
# lst_dir0.append( r"D:/FYP/data/L0.05_vf0.6/h0.0002" )
# dir_mesh = r"D:/FYP/data/mesh"

#AI LAB FILEPATH
#-------------------------------------------------------
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.15/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.25/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.35/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.45/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.55/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/unseen/L0.05_vf0.65/h0.0002" )
dir_mesh = r"/media/dg765/72EE-5033/FYP/data/unseen/mesh"

# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.25/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.3/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.5/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.6/h0.0002" )
# dir_mesh = r"/media/dg765/72EE-5033/FYP/data/mesh"

# Define the directory where the trained model will be saved
save_dir = './saved_models/'
os.makedirs(save_dir, exist_ok=True)

# Load Dataset in 251x251 resolution
train_loader, test_loaders, data_processor = load_fracture_mesh_stress(
    lst_dir0=lst_dir0, dir_mesh=dir_mesh,
    train_resolution=config.data.train_resolution,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
    num_workers=config.data.num_workers,
    pin_memory=config.data.pin_memory,
    persistent_workers=config.data.persistent_workers,
)


# Define the paths to the saved model and data
saved_model_path = './saved_models/best_damage.pt'

# Load the saved model
data_processor = data_processor.to(device)
model = get_model(config)
model.load_state_dict(torch.load(saved_model_path))
model.to(device)
model.eval()

test_samples = test_loaders[251].dataset

# After evaluating the model on the test data, compute the F1 score
f1_scores = []

y_true = []
y_pred = []

samples_to_test = 250

for index in range(samples_to_test):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    # Convert predictions to numpy array
    y_pred = out.squeeze().detach().cpu().numpy()
    # Convert ground truth to numpy array
    y_true = y.squeeze().cpu().numpy()
    
    # Define a threshold value
    threshold = 0.5
    
    # Convert continuous predictions to binary labels
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Convert continuous ground truth values to binary labels
    y_true_binary = (y_true >= threshold).astype(int)
    
    # Compute F1 score
    f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    
    f1_scores.append(f1)
print("F1:", np.mean(f1_scores))


if config.wandb.log and is_logger:
    wandb.finish()

# Plot histogram
plt.figure(figsize=(10, 6))

# Plot histogram for the channel
plt.hist(f1_scores, bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.title('F1 Scores Histogram')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')

plt.show()

fig = plt.figure(figsize=(75, 100))
fontsize = 10  # Changed to integer

samples_to_plot = 3

# Define a grid with 4 rows (3 for plots and 1 for color bars) and 3 columns
gs = GridSpec(samples_to_plot + 1, 3, height_ratios=[1]*samples_to_plot + [0.1])

# Assuming test_samples and data_processor are defined elsewhere
for index in range(samples_to_plot):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    
    # Convert the tensor to a numpy array
    y_np = y.cpu().numpy()  # Assuming y is on GPU, move it to CPU and convert to numpy array
    
    # Get the minimum and maximum values from the numpy array
    min_value = y_np.min()
    max_value = y_np.max()

    print("MINVALUE:", min_value)
    print("MAXVALUE:", max_value)

    ax = fig.add_subplot(gs[index, 0])
    im = ax.imshow(x[0].cpu().numpy())  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Input', fontsize=fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(gs[index, 1])
    im = ax.imshow(y.squeeze().cpu().numpy(), vmin=0, vmax=1, cmap='rainbow')
    if index == 0: 
        ax.set_title('Ground-truth', fontsize=fontsize)
    
    ax = fig.add_subplot(gs[index, 2])
    im = ax.imshow(out.squeeze().detach().cpu().numpy(), vmin=0, vmax=1, cmap='rainbow')  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction', fontsize=fontsize)

 # Add color bars to the bottom row for the second and third columns
    if index == samples_to_plot - 1:  # Only add color bar for the last row
        for col in range(1, 3):  # Loop through the second and third columns
            cbar_ax = fig.add_subplot(gs[samples_to_plot, col])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')  # Adjust shrink parameter
            cbar.ax.tick_params(labelsize=fontsize)  # Adjust color bar font size
            cbar_ax.set_aspect(0.2)  # Adjust the aspect ratio of the color bar


fig.suptitle('Inputs, ground-truth outputs, and predictions.',fontsize=fontsize, x=0.5, y=1.0, ha='center')
plt.tight_layout()
plt.subplots_adjust(bottom=0.05,top=0.9, wspace=0.5, hspace=0.2)
plt.show()  # Display the figure inline

fig = plt.figure(figsize=(150, 100))
fontsize = '10'

samples_to_plot = 3

for index in range(samples_to_plot):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    
    # Convert the tensor to a numpy array
    y_np = y.cpu().numpy()  # Assuming y is on GPU, move it to CPU and convert to numpy array
    
    # Get the minimum and maximum values from the numpy array
    min_value = y_np.min()
    max_value = y_np.max()

    ax = fig.add_subplot(3, 4, index*4 + 1)
    ax.imshow(x[0].cpu().numpy())  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Input', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 4, index*4 + 2)

    # Binarize ground-truth
    y_true_binary = (y.squeeze().cpu().numpy() >= threshold).astype(int)

    ax.imshow(y_true_binary, vmin=0, vmax=1, cmap=cm.gray_r)
    if index == 0: 
        ax.set_title('Ground-truth', fontsize = fontsize)
    # Add F1 score to the plot
    ax.text(0.5, 0.05, f"F1 Score: {f1_scores[index]:.2f}", ha='center', transform=ax.transAxes)

    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 4, index*4 + 3)
    
    # Binarize model prediction
    y_pred_binary = (out.squeeze().detach().cpu().numpy() >= threshold).astype(int)
    
    ax.imshow(y_pred_binary, vmin=0, vmax=1, cmap=cm.gray_r)  # Binarized output
    if index == 0: 
        ax.set_title('Binarised Model Output', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 4, index*4 + 4)
    ax.imshow(out.squeeze().detach().cpu().numpy(), vmin=min_value, vmax=max_value)  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth outputs, and binarized model predictions.', y=1, fontsize = fontsize)
plt.tight_layout()
plt.subplots_adjust(bottom=0.05,top=0.9, wspace=0.5, hspace=0.2)
plt.show()  # Display the figure inline
