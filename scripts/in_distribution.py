import argparse
import json
import logging
from scripts.two_moons import log_and_print
from util.data import *
from util.general import *
from util.methods import *
import arch
from datetime import datetime
from sys import stdout
from scripts.global_constants import *
from matplotlib import cm


def in_distribution():
  # ------------------------------------------------------------------------------------------------
  # Arguments
  # ------------------------------------------------------------------------------------------------

  config = argparse.ArgumentParser(allow_abbrev=False)

  config.add_argument("--data", type=str, choices=["cifar10", "cifar100", "mnist"], required=True)
  config.add_argument("--data_root", type=str, default=MNIST_DATA_ROOT)
  config.add_argument("--batch_size", type=int, default=200)
  config.add_argument("--val_pc", type=float,
                      default=0.15)  # must match one used for train_models.py
  config.add_argument("--workers", type=int, default=1)
  config.add_argument("--model", type=str, default="")
  config.add_argument("--model_args", type=int, nargs="+",
                    default=[])  # for mnist, hidden layer sizes
  config.add_argument("--models_root", type=str, default=DEFAULT_MODELS_ROOT)
  config.add_argument("--seed", type=int, nargs="+",
                      required=True)  # to load the corresponding model, and for reproducibility
  config.add_argument("--cuda", default=False, action="store_true")
  config.add_argument("--suff", type=str, default="")
  config.add_argument("--active",type=str,default="")
  config.add_argument("--threshold_divs", type=int, default=1000)
  config.add_argument("--mode",type=int, default=1)
  config.add_argument("--file",type=str,default="")
  subparsers = config.add_subparsers(dest="method")

  subfunctions_config = subparsers.add_parser("subfunctions")
  ensemble_subfunctions_config = subparsers.add_parser("ensemble_subfunctions")

  for subconfig in [subfunctions_config, ensemble_subfunctions_config]:
    subconfig.add_argument("--search_deltas", type=float, nargs="+",
                           default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    subconfig.add_argument("--search_ps", type=float, nargs="+", required=True)
    subconfig.add_argument("--precompute", default=False, action="store_true")
    subconfig.add_argument("--precompute_p_i", type=int, default=-1)
    subconfig.add_argument("--pattern_batch_sz", type=int,
                           default=-1)  # set to -1 to do whole dataset at once

    subconfig.add_argument("--no_bound", default=False, action="store_true")
    subconfig.add_argument("--no_log", default=False, action="store_true")
    subconfig.add_argument("--dist_fn", type=str, default="gaussian", choices=["gaussian"])
    subconfig.add_argument("--select_on_AUROC", default=False, action="store_true")

    subconfig.add_argument("--test_code_brute_force", default=False, action="store_true")

  class_distance_config = subparsers.add_parser("class_distance")
  class_distance_config.add_argument("--search_eps", type=float, nargs="+",
                                     default=[0.001, 0.005, 0.01, 0.05, 0.1])
  class_distance_config.add_argument("--balance_data", default=False, action="store_true")

  # explicit_density_config = subparsers.add_parser("explicit_density")
  # explicit_density_config.add_argument("--density_model_path_pattern", type=str,
  #                                      default=RESIDUAL_FLOWS_MODEL_PATH_PATT)

  gaussian_process_config = subparsers.add_parser("gaussian_process")
  gaussian_process_config.add_argument("--gp_hidden_dim", type=int, default=1024)
  gaussian_process_config.add_argument("--gp_scales", type=float, nargs="+", default=[1., 2., 4., 8.])

  _ = subparsers.add_parser("max_response")
  _ = subparsers.add_parser("entropy")
  _ = subparsers.add_parser("margin")
  _ = subparsers.add_parser("ensemble_var")
  _ = subparsers.add_parser("ensemble_max_response")

  tack_et_al_config = subparsers.add_parser("tack_et_al")
  tack_et_al_config.add_argument("--tack_et_al_split_batch", type=int, default=10)

  bergman_et_al_config = subparsers.add_parser("bergman_et_al")
  bergman_et_al_config.add_argument("--bergman_et_al_M", type=int, default=50)

  dropout_config = subparsers.add_parser("dropout")
  dropout_config.add_argument("--dropout_ps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
  dropout_config.add_argument("--dropout_iterations", type=int, default=10)
  
  config = config.parse_args()
  print("Config: %s" % config)

  # ------------------------------------------------------------------------------------------------
  # Script
  # ------------------------------------------------------------------------------------------------

  start_time = datetime.now()

  set_seed(config.seed[0])  # for reproducibility
  train_loader, val_loader, test_loader = get_data(config, val_pc=config.val_pc, training=False)

  for loader in [train_loader, val_loader, test_loader]:
    print_first_labels(loader)  # sanity

  model = [
    torch.load(osp.join(config.models_root, "%s_%d_%s_%s_%s.pytorch" % (config.data, s, config.model, '_'.join(map(str, config.model_args)),config.file)))[
      "model"].eval() for s in config.seed]
  acc = [
    torch.load(osp.join(config.models_root, "%s_%d_%s_%s_%s.pytorch" % (config.data, s, config.model, '_'.join(map(str, config.model_args)),config.file)))[
      "acc"] for s in config.seed]
  if len(config.seed) == 1:
    config.seed = config.seed[0]
    model = model[0]
    acc = acc[0]
    print("original acc: %s" % torch.load(
      osp.join(config.models_root, "%s_%d_%s_%s_%s.pytorch" % (config.data, config.seed, config.model, '_'.join(map(str, config.model_args)),config.file)))[
      "acc"])

  # Store precomputations if not stored already and find val data hyperparameters if any. Also
  # adapt model if necessary.
  model, method_variables = globals()["%s_pre" % config.method](config, model, train_loader,
                                                                val_loader)
  if config.mode ==1:
     path = "base"
  elif config.mode ==2:
     path = "freeze"
  elif config.mode == 3:
     path = "delete"

  log_dir = os.path.join('log', path, 'cifar10', str(config.seed),str(config.model_args),config.file)
  os.makedirs(log_dir, exist_ok=True)
  log_file = os.path.join(log_dir, f'{config.data}_{config.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

  logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'), 
            logging.StreamHandler()
        ]
    )
  # Run through test data batches, pass each batch to metric method along with needed params,
  # get metrics back, store with ground truth
  polytope_sample_counts = {}

  unreliability = []
  corrects = []
  for batch_i, (imgs, targets) in enumerate(test_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    corrects_i, polytopes_i= globals()["%s_metric" % config.method](config, method_variables,
                                                                         model, imgs, targets)
    
    # print(imgs.shape[0]) # 200 
    for j in range(imgs.shape[0]):
        polytope_str = bool_tensor_content_hash(polytopes_i[j])
        # print(f'polytope : {polytope_str}') 
        if polytope_str not in polytope_sample_counts:
            polytope_sample_counts[polytope_str] = 0
        polytope_sample_counts[polytope_str] += 1
  # print(f'counts: {len(polytope_sample_counts)}') 
  unique_counts = sorted(set(polytope_sample_counts.values()))
  count_to_color = {count: cm.viridis(i / len(unique_counts)) for i, count in enumerate(unique_counts)}

  # Convert colors to RGB format (0-255)
  count_to_color = {k: [int(c * 255) for c in v[:3]] for k, v in count_to_color.items()}

  # Assign colors based on sample counts
  polytope_colours = {polytope_str: count_to_color[count] for polytope_str, count in polytope_sample_counts.items()}
    


  for batch_i, (imgs, targets) in enumerate(test_loader):
    imgs, targets = imgs.to(device(config.cuda)), targets.to(device(config.cuda))

    corrects_i, polytopes_i= globals()["%s_metric" % config.method](config, method_variables,
                                                                         model, imgs, targets)
    # unreliability.append(unreliability_i)
    corrects.append(corrects_i)

  # unreliability = torch.cat(unreliability)
  corrects = torch.cat(corrects)
  log_and_print("seed: %d" % config.seed)
  log_and_print("polytope colours sz: %s" % len(polytope_colours))
  log_and_print("num populated polytopes: %s" % method_variables["train_ind_to_patt"].shape[0])
  log_and_print(f'sample counts:{torch.sum(method_variables["sample_counts"])}')
  log_and_print(f'corrects:{torch.sum(method_variables["corrects"])}')
  log_and_print(f'total sample: {len(train_loader.dataset)}')
  with open(config.active, 'r') as f:
        activations = json.load(f)
  for key, value in activations.items():

    average = activations[key]["Average"]
    std = activations[key]["Std"]
    log_and_print(f'Average AASR: {average}')
    log_and_print(f'std: {std}')
  # print(method_variables["train_patt_to_ind"])

  


  store_fname_prefix = "%s_%s_%s_%s_%s" % (
  config.data, config.seed, config.method, config.model, config.suff)
  # eval_and_store(config, unreliability, corrects, method_variables, store_fname_prefix)
  print("Took time: %s" % (datetime.now() - start_time))

  num_test_side = 501 
  xx, yy = np.meshgrid(np.linspace(0, num_test_side - 1, num_test_side),
                       np.linspace(0, num_test_side - 1, num_test_side))  # -1 in middle!!
  X = xx
  Y = yy
  whole_model_metric = (1. - acc)
  if not config.no_bound:
    whole_model_metric += torch.sqrt(
      torch.log(2. / method_variables["delta"]) / (2. * method_variables["m"])).item()
    if not config.no_log:
      whole_model_metric = np.log(whole_model_metric)

  Z_unreliability = whole_model_metric * np.ones(X.shape)

  log_and_print("whole_model_metric %s" % whole_model_metric)
  log_and_print("Z_unreliability %s" % np.unique(Z_unreliability))

  log_and_print("orig model acc: %s" % acc)

  total_samples = sum(polytope_sample_counts.values())
  average_samples = np.mean(list(polytope_sample_counts.values()))
  std_samples = np.std(list(polytope_sample_counts.values()))
  log_and_print(f'total samples: {total_samples}')
  log_and_print(f'average number of samples per polytope: {average_samples}')
  log_and_print(f'standard deviation: {std_samples}')



def bool_tensor_content_hash(curr_pattern):
    curr_pattern = curr_pattern.view(-1).cpu().numpy()
    return "".join([str(x) for x in curr_pattern])

if __name__ == "__main__":
  in_distribution()