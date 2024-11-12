import os
import random
from sys import stdout
import argparse

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--seed", type=int, default=4)
args.add_argument("--gpu", type=str, default="0")
args.add_argument("--radius_mult", type=float, default=3.0)
args.add_argument("--file",type=str,default="")
args.add_argument("--ts",type=float,default=0)
args.add_argument("--lr",type=float,default=0.01)
args.add_argument("--nclass",type=int,default=10)
args = args.parse_args()

seeds = [random.randint(0, 2**32 - 1) for _ in range(5)]
# seeds = [4259898581]
# seeds = [3442538909]
# seeds = [154934714]
# seeds = [514327440]
# seeds = [386431994, 929997039, 4259898581,511232505]
# seeds = [1882036893, 2813555025, 4259898581, 3654477082, 2394402426] 
# seeds = [1335057385, 154934714, 2931545256, 729027343, 907327111]
# seeds = [3230984738, 2269288743, 1170293192, 4240719806, 2018440254]
seeds = [4232357360, 3665243643, 277663570, 3442538909, 2271731698]
# seeds = [1467730072, 1635520989, 4262106640]

train_model = True
precomp = True
postcomp = True


norm_data_str = " --two_moons_norm_data "
search_ps = "1.0 2.0 4.0 6.0 8.0 12.0 16.0 24.0 32.0 48.0 64.0"
gpu_list = args.gpu.split(",")  # Split comma-separated list into individual GPU IDs
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
for i in seeds:
# i = 1

  # if train_model:
  #   train_model_cmd = "python -m scripts.train_models --data " \
  #                     "cifar10 --model wrn --seed %d --model_args 2 --file %s --lr %f --epochs 200 " \
  #                     "--cuda --batch_size 64 --workers 8 %s --mode 1 --ts %f --num_class %d"
  #   train_model_cmd = train_model_cmd % (i, args.file, args.lr, norm_data_str, args.ts,args.nclass)

  #   print("Executing training %s" % train_model_cmd)
  #   stdout.flush()
  #   os.system(train_model_cmd)



    train_model_cmd = "python -m scripts.train_models --data " \
                      "cifar100 --model vgg --seed %d --model_args 2 --file %s --lr %f --epochs 200 " \
                      "--cuda --batch_size 64 --workers 8 %s --mode 2 --ts %f --num_class %d"
    train_model_cmd = train_model_cmd % ( i, args.file, args.lr, norm_data_str, args.ts, args.nclass)

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
