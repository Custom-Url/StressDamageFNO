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

from neuralop.utils import get_wandb_api_key, count_model_params

import matplotlib.pyplot as plt

import wandb

from neuralop.models.fno import FNO2d

from sklearn.metrics import f1_score


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
    wandb.login(key="insert-wandb-key")
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno2d.n_layers,
                config.fno2d.n_modes_width,
                config.fno2d.n_modes_height,
                config.fno2d.hidden_channels,
                config.fno2d.factorization,
                config.fno2d.rank,
                config.patching.levels,
                config.patching.padding,
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
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.25/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.3/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.5/h0.0002" )
lst_dir0.append( r"/media/dg765/72EE-5033/FYP/data/L0.05_vf0.6/h0.0002" )
dir_mesh = r"/media/dg765/72EE-5033/FYP/data/mesh"

# Define the directory where the trained model will be saved
save_dir = './saved_models/'
os.makedirs(save_dir, exist_ok=True)

# Define the best validation loss
best_f1 = float('-inf')
best_model_path = os.path.join(save_dir, 'best_stress.pt')

# Define sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "f1", "goal": "maximize"},
    "parameters": {}
}

# Add parameters from the config file to the sweep configuration
for param_name, param_values in config.wandb.params.items():
    sweep_config["parameters"][param_name] = {"values": param_values["values"]}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config, project="Stress")

def train():
    wandb.init(project="Stress")
    cfg = wandb.config

    # Update config values with those stored in cfg
    config.data.n_train = cfg.n_train
    config.opt.n_epochs = cfg.n_epochs
    config.opt.learning_rate = cfg.learning_rate
    config.opt.weight_decay = cfg.weight_decay

    config.fno2d.data_channels = config.fno2d.in_channels

    run_name = "_".join(
    f"{var}"
    for var in [
        "Stress",
        config.opt.n_epochs,
        config.data.n_train,
        config.opt.learning_rate,
        config.opt.weight_decay,
    ]
    )  
    wandb.run.name = run_name 

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

    print("CONFIG:", config)

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

    # only perform MG patching if config patching levels > 0

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


        # Log F1 score to WandB
    if config.wandb.log and is_logger:
        wandb.log({"learning_rate": config.opt.learning_rate})
        wandb.log({"weight_decay": config.opt.weight_decay})
        wandb.log({"n_epochs": config.opt.n_epochs})
        wandb.log({"n_train": config.data.n_train})
        wandb.log({"f1": np.mean(f1_scores)})
            
        # Print logged parameters
        print("Logged Parameters:")
        print("Learning Rate:", config.opt.learning_rate)
        print("Weight Decay:", config.opt.weight_decay)
        print("Number of Epochs:", config.opt.n_epochs)
        print("Number of Training Samples:", config.data.n_train)
        print("F1 Score:", np.mean(f1_scores))
        print("RUN COMPLETE")

    # Check if current model has the best F1 score
    global best_f1
    if np.mean(f1_scores) > best_f1:
        print("PREVIOUS F1:", best_f1)
        best_f1 = np.mean(f1_scores)
        print("NEW F1:", best_f1)
        # Save the current best model
        torch.save(model.state_dict(), best_model_path)

# Run the sweep
wandb.agent(sweep_id, function=train)


if config.wandb.log and is_logger:
    wandb.finish()




