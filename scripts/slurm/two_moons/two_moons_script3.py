import os
import random
from sys import stdout
import argparse

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--seed", type=int, default=4)
args.add_argument("--gpu", type=int, default=5)
args.add_argument("--radius_mult", type=float, default=3.0)

args = args.parse_args()

seeds = [random.randint(0, 2**32 - 1) for _ in range(5)]

train_model = True
precomp = True
postcomp = True

norm_data_str = " --two_moons_norm_data "
search_ps = "1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"

for i in seeds:
# i = 1

  if train_model:
    train_model_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.train_models3 --data " \
                      "cifar10 --model mlp --model_args 8 8  --seed %d --lr 0.1 --epochs 200 " \
                      "--cuda --batch_size 100 --workers 8 %s --mode 1 --seed %d"
    train_model_cmd = train_model_cmd % (args.gpu, args.seed, norm_data_str,i)

    print("Executing training %s" % train_model_cmd)
    stdout.flush()
    os.system(train_model_cmd)



    train_model_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.train_models3 --data " \
                      "cifar10 --model mlp --model_args 8 8  --seed %d --lr 0.1 --epochs 200 " \
                      "--cuda --batch_size 100 --workers 8 %s --mode 2 --seed %d"
    train_model_cmd = train_model_cmd % (args.gpu, args.seed, norm_data_str,i)

    print("Executing training %s" % train_model_cmd)
    stdout.flush()
    os.system(train_model_cmd)

# if precomp:
#   precomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data mnist " \
#                 "--model mlp --seed %d --cuda --batch_size 50 --workers 1 %s subfunctions " \
#                 "--search_ps 1.0 --pattern_batch_sz 1000 --precompute --precompute_p_i 0"

#   for p_i in range(11):
#     precomp_cmd = precomp_str % (args.gpu, args.seed, norm_data_str, search_ps, p_i)

#     print("Executing precomp %s" % precomp_cmd)
#     stdout.flush()
#     os.system(precomp_cmd)

# if postcomp:
#   postcomp_str = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.two_moons --data mnist " \
#                  "--model mlp --seed %d --cuda --batch_size 100 --workers 1 %s --radius_mult %s " \
#                  "subfunctions --search_ps %s"
#   postcomp_cmd = postcomp_str % (args.gpu, args.seed, norm_data_str, args.radius_mult, search_ps)
#   print("Executing postcomp %s" % postcomp_cmd)
#   stdout.flush()
#   os.system(postcomp_cmd)
