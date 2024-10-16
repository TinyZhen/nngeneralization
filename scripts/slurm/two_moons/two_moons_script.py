import os
import random
from sys import stdout
import argparse

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--seed", type=int, default=4)
args.add_argument("--gpu", type=int, default=3)
args.add_argument("--radius_mult", type=float, default=3.0)
args.add_argument("--file",type=str,default="")
args.add_argument("--ts",type=float,default=0)
args = args.parse_args()

seeds = [random.randint(0, 2**32 - 1)]
# seeds = [4259898581]
# seeds = [3442538909]
# seeds = [386431994]
# seeds = [386431994, 929997039, 4259898581,511232505]

train_model = True
precomp = True
postcomp = True


norm_data_str = " --two_moons_norm_data "
search_ps = "1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"

for i in seeds:
# i = 1

  if train_model:
    train_model_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.train_models --data " \
                      "cifar10 --model vgg --seed %d --model_args 1 --file %s --lr 0.1 --epochs 200 --gpu %d " \
                      "--cuda --batch_size 32 --workers 8 %s --mode 1 --ts %f"
    train_model_cmd = train_model_cmd % (args.gpu, i, args.file, args.gpu, norm_data_str, args.ts)

    print("Executing training %s" % train_model_cmd)
    stdout.flush()
    os.system(train_model_cmd)



    # train_model_cmd = "export CUDA_VISIBLE_DEVICES=%d && python -m scripts.train_models --data " \
    #                   "cifar10 --model vgg --seed %d --model_args 1 --file %s --lr 0.1 --epochs 200 --gpu %d " \
    #                   "--cuda --batch_size 32 --workers 8 %s --mode 2 --ts %f"
    # train_model_cmd = train_model_cmd % (args.gpu, i, args.file, args.gpu, norm_data_str, args.ts)

    # print("Executing training %s" % train_model_cmd)
    # stdout.flush()
    # os.system(train_model_cmd)

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
