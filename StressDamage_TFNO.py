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

from utils.StressDamage_Load import load_fracture_mesh_stress

from neuralop.utils import count_model_params

import matplotlib.pyplot as plt

import wandb

from neuralop.models.fno import FNO2d

from sklearn.metrics import f1_score


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./StressDamage_Config.yaml", config_name="default", config_folder="../dg765/config"
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
                "StressDamage",
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
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.25/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.3/h0.0002" )
# lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.5/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.6/h0.0002" )
dir_mesh = r"/media/dg765/72EE-5033/FYP/data/mesh"

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



train_dataset =train_loader.dataset

for res, test_loader in test_loaders.items():
    print(res)
    batch = next(iter(test_loader))
    x = batch['x']
    y = batch['y']
    
    print(f'Testing samples for res {res} have shape {x.shape[1:]}')
    
data = train_dataset[0]
x = data['x']
y = data['y']

print(f'Training samples for res {res} have shape {x.shape[1:]}')


# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             positional_encoding=data_processor.positional_encoding,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels)

data_processor = data_processor.to(device)
model = get_model(config)
model = model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

callbacks = [
    BasicLoggerCallback(wandb_init_args)
]

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    callbacks=callbacks,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)


trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
   
test_samples = test_loaders[251].dataset
# Move model to device
model.to(device)

# Use model.eval() for evaluation mode
model.eval()

# After evaluating the model on the test data, compute the F1 score
f1_scores = []
yy_scores = []
xx_scores = []
xy_scores = []
dmg_scores = []

y_true = []
y_pred = []

samples_to_test = 100

for index in range(samples_to_test):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))
    for channel in range(4):
        # Convert predictions to numpy array
        y_pred = out.squeeze()[channel].detach().cpu().numpy()
        # Convert ground truth to numpy array
        y_true = y.squeeze()[channel].cpu().numpy()
        
        # Define a threshold value
        threshold = 0.5
        
        # Convert continuous predictions to binary labels
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Convert continuous ground truth values to binary labels
        y_true_binary = (y_true >= threshold).astype(int)
        
        # Compute F1 score
        f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        
        if channel == 0:
            yy_scores.append(f1)
        if channel == 1:
            xx_scores.append(f1)
        if channel == 2:
            xy_scores.append(f1)
        if channel == 3:
            dmg_scores.append(f1)
        # Append F1 score to the list
        f1_scores.append(f1)
print("F1:", np.mean(f1_scores))
print("yy_score:", np.mean(yy_scores))
print("xx_score:", np.mean(xx_scores))
print("xy_score:", np.mean(xy_scores))
print("dmg_score:", np.mean(dmg_scores))


if config.wandb.log and is_logger:
    wandb.finish()

fig = plt.figure(figsize=(150, 75))
fontsize = '10'

samples_to_plot = 3
min_value = [0.0] * 4
max_value = [0.0] * 4

for index in range(samples_to_plot):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Move input tensor to device
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0))

    for channel in range(4):
        # Convert the tensor to a numpy array
        y_np = y.squeeze()[channel].cpu().numpy()  # Assuming y is on GPU, move it to CPU and convert to numpy array
        # Get the minimum and maximum values from the numpy array
        min_value[channel] = y_np.min()
        max_value[channel] = y_np.max()
    print("MINVALUE:", min_value)
    print("MAXVALUE:", max_value)

    ax = fig.add_subplot(3, 9, index*9 + 1)
    ax.imshow(x[0].cpu().numpy())  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Input x', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 9, index*9 + 2)
    ax.imshow(y.squeeze()[0].cpu().numpy(), vmin=min_value[0], vmax=max_value[0])
    if index == 0: 
        ax.set_title('Ground-truth yy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 3)
    ax.imshow(out.squeeze()[0].detach().cpu().numpy(), vmin=min_value[0], vmax=max_value[0])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction yy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 4)
    ax.imshow(y.squeeze()[1].cpu().numpy(), vmin=min_value[1], vmax=max_value[1])
    if index == 0: 
        ax.set_title('Ground-truth xx', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 5)
    ax.imshow(out.squeeze()[1].detach().cpu().numpy(), vmin=min_value[1], vmax=max_value[1])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction xx', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 6)
    ax.imshow(y.squeeze()[2].cpu().numpy(), vmin=min_value[2], vmax=max_value[2])
    if index == 0: 
        ax.set_title('Ground-truth xy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 7)
    ax.imshow(out.squeeze()[2].detach().cpu().numpy(), vmin=min_value[2], vmax=max_value[2])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction xy', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 9, index*9 + 8)
    ax.imshow(y.squeeze()[3].cpu().numpy(), vmin=min_value[3], vmax=max_value[3])
    if index == 0: 
        ax.set_title('Ground-truth Damage', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])
    
    ax = fig.add_subplot(3, 9, index*9 + 9)
    ax.imshow(out.squeeze()[3].detach().cpu().numpy(), vmin=min_value[3], vmax=max_value[3])  # Convert to CPU for visualization
    if index == 0: 
        ax.set_title('Model prediction Damage', fontsize = fontsize)
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth outputs, and predictions.', y=0.98, fontsize = fontsize)
plt.tight_layout()
plt.show()  # Display the figure inline



"""
# Create the trainer

save_dir = f'../model/{save_folder}/R{RES}_nmodes{nmodes}_liftch{liftchannels}_nlayers{nlayers}_fourierch{fourierchannels}_projch{projchannels}/'

os.makedirs(save_dir, exist_ok=True)
 
trainer = Trainer(model=model, n_epochs=100,

                  device=device,

                  callbacks=[

                      CheckpointCallback(save_dir=save_dir,

                                         save_best=False,

                                         save_interval=10,

                                         save_optimizer=True,

                                         save_scheduler=True),

                      BasicLoggerCallback(),

                        ],

                  data_processor=data_processor,

                  wandb_log=False,

                  log_test_interval=3,

                  use_distributed=False,

                  verbose=True)

# load the trained model

model.load_state_dict(torch.load(save_dir+"/model_state_dict.pt"))
 

"""


"""
scipy.ndimage.zoom

order=0 for the input, order=1 for the outputs
"""
