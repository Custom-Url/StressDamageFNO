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

from utils.LOAD_fracture_Stress import load_fracture_mesh_stress

from neuralop.utils import count_model_params

import matplotlib.pyplot as plt

import wandb

from neuralop.models.fno import FNO2d


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./Elastic_TFNO_config.yaml", config_name="default", config_folder="../dg765/config"
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
    wandb.login(key="4e31870235d5cf5bcbe8fa12a916362a75a51d9e")
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
saved_model_path = './saved_models/0.7_stress.pt'

# Load the saved model
data_processor = data_processor.to(device)
model = get_model(config)
model.load_state_dict(torch.load(saved_model_path))
model.to(device)
model.eval()

test_samples = test_loaders[251].dataset

f1_scores = []

samples_to_test = 1000

for index in range(samples_to_test):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    # Convert to numpy array
    mesh = x.squeeze().cpu().numpy()
    true = y.squeeze().cpu().numpy()
    pred = out.squeeze().detach().cpu().numpy()
    
    # Compute I1 for both ground truth and prediction
    I1 = (true[0] + true[1] + true[2])
    I1_p = (pred[0] + pred[1] + pred[2])
    # Compute segmented areas
    true_hot = I1 > np.percentile(I1[mesh[0, 0] != 3], 99)
    pred_hot = I1_p > np.percentile(I1_p[mesh[0, 0] != 3], 99)
    # Calculate True Positives, False Positives, False Negatives
    TP = np.count_nonzero(pred_hot & true_hot)
    FP = np.count_nonzero(pred_hot & ~true_hot)
    FN = np.count_nonzero(~pred_hot & true_hot)
    # Compute F1 score
    F1 = 2 * TP / (2 * TP + FP + FN)
    f1_scores.append(F1)
print("F1:", np.mean(f1_scores))


if config.wandb.log and is_logger:
    wandb.finish()
# Plot histogram
plt.figure(figsize=(10, 6))

# Plot histogram for the channel
plt.hist(f1_scores, bins=10, color='skyblue', edgecolor='black')

plt.xlim(0, 1)

# Add labels and title
plt.title('F1 Scores Histogram')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')

plt.show()

fig = plt.figure(figsize=(150, 75))
fontsize = '10'

samples_to_plot = 3
min_value = [0.0] * 3
max_value = [0.0] * 3

for index in range(samples_to_plot):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))

    for channel in range(3):
        # Convert the tensor to a numpy array
        y_np = y.squeeze()[channel].cpu().numpy()  # Assuming y is on GPU, move it to CPU and convert to numpy array
        # Get the minimum and maximum values from the numpy array
        min_value[channel] = y_np.min()
        max_value[channel] = y_np.max()

    ax = fig.add_subplot(3, 7, index*7 + 1)
    ax.imshow(x[0].cpu().numpy())  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Input x', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 7, index*7 + 2)
    ax.imshow(y.squeeze()[0].cpu().numpy(), vmin=min_value[0], vmax=max_value[0])
    if index == 0: 
        ax.set_title('Ground-truth yy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 7, index*7 + 3)
    ax.imshow(out.squeeze()[0].detach().cpu().numpy(), vmin=min_value[0], vmax=max_value[0])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction yy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 7, index*7 + 4)
    ax.imshow(y.squeeze()[1].cpu().numpy(), vmin=min_value[1], vmax=max_value[1])
    if index == 0: 
        ax.set_title('Ground-truth xx', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 7, index*7 + 5)
    ax.imshow(out.squeeze()[1].detach().cpu().numpy(), vmin=min_value[1], vmax=max_value[1])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction xx', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 7, index*7 + 6)
    ax.imshow(y.squeeze()[2].cpu().numpy(), vmin=min_value[2], vmax=max_value[2])
    if index == 0: 
        ax.set_title('Ground-truth xy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 7, index*7 + 7)
    ax.imshow(out.squeeze()[2].detach().cpu().numpy(), vmin=min_value[2], vmax=max_value[2])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction xy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])


fig.suptitle('Inputs, ground-truth outputs, and predictions.', y=0.98, fontsize = fontsize)
plt.tight_layout()
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
    
    # Convert to numpy array
    mesh = x.squeeze().cpu().numpy()
    true = y.squeeze().cpu().numpy()
    pred = out.squeeze().detach().cpu().numpy()
    
    # Compute I1 for both ground truth and prediction
    I1 = (true[0] + true[1] + true[2])
    I1_p = (pred[0] + pred[1] + pred[2])
    # Compute segmented areas
    true_hot = I1 > np.percentile(I1[mesh[0, 0] != 3], 99)
    pred_hot = I1_p > np.percentile(I1_p[mesh[0, 0] != 3], 99)
    
    ax = fig.add_subplot(samples_to_plot, 3, index * 3 + 1)
    ax.imshow(x[0].cpu().numpy())  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Input', fontsize=fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    # Add F1 score label
    ax.text(0, 0, f"F1 Score: {f1_scores[index]}", color='red', fontsize=12, transform=ax.transAxes)

    ax = fig.add_subplot(samples_to_plot, 3, index * 3 + 2)
    ax.imshow(true_hot, cmap='binary')
    if index == 0: 
        ax.set_title('True Hot', fontsize=fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(samples_to_plot, 3, index * 3 + 3)
    ax.imshow(pred_hot, cmap='binary')
    if index == 0: 
        ax.set_title('Predicted Hot', fontsize=fontsize)
    plt.xticks([], [])
    plt.yticks([], [])


fig.suptitle('Inputs, true hot areas, predicted hot areas, and model predictions', y=1, fontsize=fontsize)
plt.tight_layout()
plt.show()


print("MINVALUE:", min_value)
print("MAXVALUE:", max_value)