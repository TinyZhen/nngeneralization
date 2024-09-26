import argparse
import json
import logging
import os.path as osp
import torch
from util.data import *
from util.general import *
from torch.nn import CrossEntropyLoss
from torch import optim
from datetime import datetime
from sys import stdout
from functools import partial
from scripts.global_constants import *
import matplotlib.pyplot as plt
import arch


norm_data_str = " --two_moons_norm_data "
search_ps = "1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"

config = argparse.ArgumentParser(allow_abbrev=False)
config.add_argument("--data", type=str, choices=["mnist", "cifar10", "cifar100", "two_moons"],
                    required=True)
config.add_argument("--data_root", type=str, default=CIFAR_DATA_ROOT)
config.add_argument("--batch_size", type=int, default=200)
config.add_argument("--workers", type=int, default=1)

config.add_argument("--models_root", type=str, default=DEFAULT_MODELS_ROOT)
config.add_argument("--model_args", type=int, nargs="+",
                    default=[])  # for mnist, hidden layer sizes
config.add_argument("--seed", type=int, default=0)
config.add_argument("--lr", type=float, default=0.1)
config.add_argument("--weight_decay", type=float, default=5e-4)
config.add_argument("--epochs", type=int, default=100)
config.add_argument("--gpu", type=int, default=0)

config.add_argument("--lr_sched_epoch_gap", type=int, default=30)
config.add_argument("--lr_sched_mult", type=float, default=0.1)

config.add_argument("--cuda", default=False, action="store_true")
config.add_argument("--restart", default=False, action="store_true")

config.add_argument("--model", type=str, required=True)
config.add_argument("--val_pc", type=float, default=0.15)

config.add_argument("--two_moons_norm_data", default=False, action="store_true")
config.add_argument("--mode",type=int,default=1)
config = config.parse_args()

if not osp.exists(osp.join(config.data_root, "%s_stats.pytorch" % config.data)):
  compute_dataset_stats(config)

save_fname = osp.join(config.models_root,
                      "%s_%d_%s_%s.pytorch" % (config.data, config.seed, config.model, '_'.join(map(str, config.model_args))))


set_seed(config.seed)

train_loader, val_loader, test_loader = get_data(config, val_pc=config.val_pc,
                                                 training=True)  # same as prediction filtering
# data loading incl val pc
if config.mode ==1:
    path = "base"
elif config.mode ==2:
    path = "freeze"
elif config.mode == 3:
    path = "delete"

log_dir = os.path.join('log', path, 'cifar10', str(config.seed),str(config.model_args))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'{config.data}_{config.model}_loss.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'), 
        logging.StreamHandler()
    ]
)
def log_and_print(msg):
    print(msg)
    logging.info(msg)

print("Config: %s" % config)

num_classes = classes_per_dataset[config.data]

if config.data == "cifar10" or config.data == "cifar100":
#   if config.model == "vgg16_bn":
#     model = arch.vgg16_bn(num_classes)
#   elif config.model == "resnet50model":
#     model = arch.resnet50model(num_classes)

# if config.data == "mnist" and config.model == "mlp":
  model = arch.MLP(layer_szs=config.model_args, num_classes=num_classes, in_feats=(32 * 32 * 3))

# el
# if config.data == "two_moons" and config.model == "mlp":
#   model = arch.MLP(layer_szs=config.model_args, num_classes=num_classes, in_feats=2)
#   from util.two_moons import render_two_moons

#   render_two_moons(config, [train_loader, test_loader])

else:
  raise NotImplementedError


accs = []
next_ep = 0
def count_activation_frequency(model, loader):
    for param in model.parameters():
        param.requires_grad = True
    activations = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            activations[name] = np.zeros(layer.weight.shape[0])

    def forward_hook(module, input, output, name):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            activations[name] += (output.detach().cpu().numpy() > 0).sum(axis=0)

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(layer.register_forward_hook(partial(forward_hook, name=name)))

    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(next(model.parameters()).device)
            model(inputs)

    for hook in hooks:
        hook.remove()

    for key, value in activations.items():
        activations[key] = value / len(loader.dataset)
        # log_and_print(f'AASR: {activations[key]}')


    return activations

def count_activation_frequency_first_layer(model, loader):
    for param in model.parameters():
        param.requires_grad = True
    activations = {}
    first_layer_found = False
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            activations[name] = np.zeros(layer.weight.shape[0])
            # first_layer_found = True

    first_layer_found = False
    def forward_hook(module, input, output, name):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            activations[name] += (output.detach().cpu().numpy() > 0).sum(axis=0)

    first_layer_found = False
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and not first_layer_found:
            hooks.append(layer.register_forward_hook(partial(forward_hook, name=name)))
            first_layer_found = True 

    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(next(model.parameters()).device)
            model(inputs)

    for hook in hooks:
        hook.remove()

    for key, value in activations.items():
        activations[key] = value / len(loader.dataset)
        # log_and_print(f'AASR: {activations[key]}')

    return activations


def calculate_threshold(frequencies, alpha):
    return np.percentile(frequencies, alpha * 100)

def mark_nodes_for_reinitialization(activations, alpha):
    mem = {}
    for layer_name, freqs in activations.items():
        threshold = calculate_threshold(freqs, alpha)
        mem[layer_name] = np.where(freqs > threshold)[0].tolist()
    return mem

def mark_nodes_for_deletion(activations, threshold):
    mem = {}
    for layer_name, freqs in activations.items():
        mem[layer_name] = np.where(freqs < threshold)[0].tolist()  
    return mem

def get_layer_sizes(model):
    layer_szs = []
    i = 0
    # print(model)
    while True:
        layer_attr_name = f"layer{i}"
        if hasattr(model, layer_attr_name):
            layer = getattr(model, layer_attr_name)
            out_features = layer.weight.shape[0]
            print(f"Layer {i}: {layer_attr_name}, out_features: {out_features}")
            print(f"Layer {i} weights: {layer.weight}")

            layer_szs.append(out_features)
            i += 1
        else:
            break
    
    return layer_szs

def delete_nodes(model, mem):
    with torch.no_grad():
        new_layer_szs = []
        layer_szs = get_layer_sizes(model)
        print(f'old layer size:{layer_szs}')
        new_layer_szs = calculate_new_layer_sizes(model, mem, layer_szs,3072)
        print(new_layer_szs)
        new_model = arch.MLP(layer_szs=new_layer_szs, num_classes=10, in_feats=3072)
        new_model = copy_weights(model, new_model,mem,new_layer_szs)
        return new_model

def calculate_new_layer_sizes(old_model, mem, layer_szs, in_feats):
    new_layer_szs = []
    previous_layer_size = in_feats
    
    for i, layer_size in enumerate(layer_szs):
        layer_name = f"layer{i}"
        if layer_name in mem:
            mask = torch.ones(layer_size, dtype=torch.bool)
            mask[mem[layer_name]] = False
            
            # Calculate the new size
            new_layer_size = mask.sum().item()
            new_layer_szs.append(new_layer_size)
            
            previous_layer_size = new_layer_size
        else:
            new_layer_szs.append(layer_size)
            previous_layer_size = layer_size
    
    return new_layer_szs

def print_parameter_shapes(model, name="Model"):
    print(f"Parameters of {name}:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def copy_weights(old_model, new_model, mem, layer_szs):
    with torch.no_grad():
        previous_layer_size = 3072  # assuming input features are fixed at 784 for MNIST

        for i, nf in enumerate(layer_szs):
            # Create a mask to keep the nodes
            old_layer = getattr(old_model, f"layer{i}")
            new_layer = getattr(new_model, f"layer{i}")

            if f'layer{i}' in mem:
                # Ensure delete_mask is a long tensor
                delete_mask = torch.tensor(mem[f'layer{i}'], dtype=torch.long)
                keep_mask = torch.ones(old_layer.weight.shape[0], dtype=torch.bool)
                
                # Set indices in keep_mask to False based on delete_mask
                keep_mask[delete_mask] = False
                
                # Apply the keep_mask to the current layer's output features
                new_layer.weight.data = old_layer.weight.data[keep_mask, :previous_layer_size]
                new_layer.bias.data = old_layer.bias.data[keep_mask]

                # Update the previous_layer_size to match the masked out_features
                previous_layer_size = new_layer.weight.shape[0]

                # If not the last layer, apply the keep_mask to the next layer's input features
                if i + 1 < len(layer_szs):
                    next_old_layer = getattr(old_model, f"layer{i + 1}")
                    next_new_layer = getattr(new_model, f"layer{i + 1}")

                    # Apply the keep_mask to the next layer's input features
                    next_new_layer.weight.data = next_old_layer.weight.data[:, keep_mask]
            else:
                # If no deletion mask, copy directly
                new_layer.weight.data = old_layer.weight.data
                new_layer.bias.data = old_layer.bias.data

        # Copy the weights of the last layer
        new_model.last_layer.weight.data = old_model.last_layer.weight.data[:, :previous_layer_size]
        new_model.last_layer.bias.data = old_model.last_layer.bias.data

        return new_model





def freeze_nodes_first_layer(model, mem):
    first_layer_found = False
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if not first_layer_found and layer_name in mem and len(mem[layer_name]) > 0:
                for idx in mem[layer_name]:
                    param.data[idx] = param.data[idx].detach().clone().requires_grad_(False)
                first_layer_found = True
            
    return model

def freeze_nodes_second_layer(model, mem):
    layer_count = 0  
    
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        
        if layer_name in mem and len(mem[layer_name]) > 0:
            
            if layer_count == 1:
                for idx in mem[layer_name]:
                    param.data[idx] = param.data[idx].detach().clone().requires_grad_(False)
                break  
        layer_count += 1  
            
    return model

def generate_graphs(config, model, train_loader, val_loader, activations):
#   config.gpu = 3
#   config.seed = 4
  config.radius_mult = 3.0
#   precomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data two_moons " \
#                   "--model mlp --seed %d --cuda --batch_size 100 --workers 1 %s subfunctions " \
#                   "--search_ps %s --precompute --precompute_p_i %d"


#   for p_i in range(11):
#     precomp_cmd = precomp_str % (config.gpu, config.seed, norm_data_str, search_ps, p_i)

#     print("Executing precomp %s" % precomp_cmd)
#     stdout.flush()
#     os.system(precomp_cmd)

#   postcomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data two_moons " \
#                  "--model mlp --seed %d --cuda --batch_size 100 --workers 1 %s --radius_mult %s " \
#                  "subfunctions --search_ps %s"
#   postcomp_cmd = postcomp_str % (config.gpu, config.seed, norm_data_str, config.radius_mult, search_ps)
#   print("Executing postcomp %s" % postcomp_cmd)
#   stdout.flush()
#   os.system(postcomp_cmd)
  aasr = {}
  for key, value in activations.items():
    average = np.average(activations[key])
    std = np.std(activations[key])
    aasr[key] = {
        "Average": average, 
        "Std": std
    }

  with open('activations.json', 'w') as f:
    json.dump(aasr, f)

  test_code_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.in_distribution --data " \
    "cifar10 --data_root %s --seed %d --model mlp --cuda --mode %d --model_args %s --active activations.json subfunctions --search_ps 1.0 " \
    "--precompute --precompute_p_i 0 --pattern_batch_sz 500 " \
    % (config.gpu,CIFAR_DATA_ROOT,config.seed,config.mode, ' '.join(map(str, config.model_args))) 
  
                    
  print(test_code_cmd)
  os.system(test_code_cmd)
  test_code_cmd ="export CUDA_VISIBLE_DEVICES=%d && python -m scripts.in_distribution --data "\
    "cifar10 --data_root %s --seed %d --model mlp --cuda --mode %d --model_args %s --active activations.json subfunctions --search_ps 1.0 " \
    "--pattern_batch_sz 500 " \
    %(config.gpu,CIFAR_DATA_ROOT,config.seed,config.mode, ' '.join(map(str, config.model_args))) 
  
  print(test_code_cmd)
  os.system(test_code_cmd)


def unfreeze_all_nodes(model):
    for param in model.parameters():
        param.requires_grad = True

# Function to track validation accuracy
def track_validation_accuracy(validation_accuracies, new_accuracy, patience):
    validation_accuracies.append(new_accuracy)
    if len(validation_accuracies) > patience:
        validation_accuracies.pop(0)
    return validation_accuracies

# Function to check if validation accuracy has improved
def validation_accuracy_not_improving(validation_accuracies, patience):
    if len(validation_accuracies) < patience:
        return False
    return all(validation_accuracies[i] >= validation_accuracies[i + 1] for i in range(patience - 1))

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []

device = torch.device(f"cuda:0"  if config.cuda else "cpu")
model.to(device).train()
opt = optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=config.weight_decay)

lr_sched = partial(lr_sched_maker, config.lr_sched_epoch_gap, config.lr_sched_mult)
sched = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lr_sched)

criterion = CrossEntropyLoss().to(device)

if config.restart:
    saved = torch.load(save_fname)

    # Check if `saved["model"]` is a model object (torch.nn.Module), then extract the state dict
    if isinstance(saved["model"], torch.nn.Module):
        # Extract the state dict from the saved model object
        saved_state_dict = saved["model"].state_dict()
    else:
        saved_state_dict = saved["model"]

    # Now load the state dict into your model
    model.load_state_dict(saved_state_dict)
    # saved = torch.load(save_fname)
    # model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["opt"])
    sched.load_state_dict(saved["sched"])
    accs = saved["accs"]
    next_ep = saved["next_ep"]
    if len(accs) > 0:
        if accs[-1][0] == next_ep:
            print(f"Resuming from epoch {next_ep}")
            accs = accs[:-1]
    else:
        print("Warning: 'accs' list is empty. Unable to access previous accuracies.")
    
    print("restarting from saved: ep %d" % next_ep)

torch.save({"model": model.state_dict(), "next_ep": next_ep, "accs": accs, "opt": opt.state_dict(), "sched": sched.state_dict()},
           save_fname)

for ep in range(next_ep, config.epochs):
    print("epoch %d %s, lr %f" % (ep, datetime.now(), opt.param_groups[0]["lr"]))

    # print(model)
    epoch_train_loss = 0
    for batch_i, (imgs, targets) in enumerate(train_loader):
        opt.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)

        preds = model(imgs)  # no softmax
        loss = criterion(preds, targets)

        loss.backward()
        opt.step()

        epoch_train_loss += loss.item()

    # Compute the average loss over all batches in the epoch
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()  # Set model to evaluation mode
    epoch_test_loss = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            epoch_test_loss += loss.item()

    epoch_test_loss /= len(test_loader)
    test_losses.append(epoch_test_loss)

    model.train() 
    log_and_print(f'train loss: {epoch_train_loss}, test loss: {epoch_test_loss}')
    sched.step()
    final_acc = evaluate(config, model, test_loader)
    torch.save({"model": model, "acc": final_acc, "accs": accs,
            "next_ep": ep, "opt": opt.state_dict(), "sched": sched.state_dict()}, save_fname)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, Shape: {param.shape}")  
    print(f'acc:{final_acc}')      
    # # freeze nodes every 10 epoch
    if (ep+1) % 1 == 0:
        if config.mode ==1: 
            activations = count_activation_frequency(model, train_loader)
            print(activations)
            generate_graphs(config, model, train_loader, val_loader,activations)

        elif config.mode ==2: 
            activations = count_activation_frequency(model, train_loader)
            mem = mark_nodes_for_reinitialization(activations, 0.5)
            model = freeze_nodes_second_layer(model, mem)
            acc = evaluate(config, model, test_loader)
            final_acc = evaluate(config, model, test_loader)
            torch.save({"model": model, "acc": final_acc, "accs": accs,
                "next_ep": ep, "opt": opt.state_dict(), "sched": sched.state_dict()}, save_fname)
            print(final_acc)
            accs.append((ep, final_acc))
            generate_graphs(config, model, train_loader, val_loader,activations)

        elif config.mode ==3: 
            # delete nodes after 30 epochs and every 10 epochs    
            if (ep+1) >= 30:
                    activations = count_activation_frequency(model, train_loader)
                    mem = mark_nodes_for_deletion(activations, 0.05)
                    print(f'old model:{model}')

                    model = delete_nodes(model, mem)
                    model.to(device)
                    print(f'new model:{model}')
                    acc = evaluate(config, model, test_loader)
                    print(acc)
                    # accs.append((ep, acc))
                    generate_graphs(config, model, train_loader, val_loader,activations)

    stdout.flush()
# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.show()

plt.savefig('training_loss_0.05.png')



#  draw for the baseline
# Train
# model.to(device(config.cuda)).train()
# opt = optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=config.weight_decay)

# lr_sched = partial(lr_sched_maker, config.lr_sched_epoch_gap, config.lr_sched_mult)
# sched = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lr_sched)

# criterion = CrossEntropyLoss().to(device(config.cuda))

# if config.restart:
#   saved = torch.load(save_fname)
#   model = saved["model"]
#   opt = saved["opt"]
#   sched = saved["sched"]
#   accs = saved["accs"]
#   next_ep = saved["next_ep"]
#   if accs[-1][0] == next_ep:
#     print("trimming stored accs")
#     accs = accs[:-1]
#   print("restarting from saved: ep %d" % next_ep)

# torch.save({"model": model, "next_ep": next_ep, "accs": accs, "opt": opt, "sched": sched},
#            save_fname)
# patience = 5
# validation_accuracies = []
# for ep in range(next_ep, config.epochs):
#   print("epoch %d %s, lr %f" % (ep, datetime.now(), opt.param_groups[0]["lr"]))
#   if ep % 10 == 0:
#     torch.save({"model": model, "next_ep": ep, "accs": accs, "opt": opt, "sched": sched},
#                save_fname)

#     acc = evaluate(config, model, test_loader)
#     print(acc)
#     accs.append((ep, acc))
#   stdout.flush()

#   for batch_i, (imgs, targets) in enumerate(train_loader):
#     opt.zero_grad()
#     imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

#     preds = model(imgs)  # no softmax
#     loss = criterion(preds, targets)

#     loss.backward()
#     opt.step()
#   activations = count_activation_frequency(model, train_loader)
#   mem = mark_nodes_for_reinitialization(activations, 0.8)
#   model = freeze_nodes(model, mem)
#   sched.step()
#   val_acc = evaluate(config, model, val_loader)
#   validation_accuracies = track_validation_accuracy(validation_accuracies, val_acc, patience)
  
#   if validation_accuracy_not_improving(validation_accuracies, patience):
#       print(f"Validation accuracy not improving for {patience} epochs. Unfreezing all nodes.")
#       unfreeze_all_nodes(model)
#   final_acc = evaluate(config, model, test_loader)
#   torch.save({"model": model, "acc": final_acc, "accs": accs,
#         "next_ep": ep, "opt": opt.state_dict(), "sched": sched.state_dict()}, save_fname)
#   generate_graphs(config, model, train_loader, val_loader)
# # Save
# final_acc = evaluate(config, model, test_loader)
# accs.append((config.epochs, final_acc))
# print("all accs %s" % accs)
# torch.save({"model": model, "acc": final_acc, "accs": accs,
#             "next_ep": ep, "opt": opt, "sched": sched}, save_fname)
