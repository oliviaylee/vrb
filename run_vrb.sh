#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=64G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="vrb" # Name the job
#SBATCH --output=clusteroutput%j.out
#SBATCH --exclude=iris[1-4],iris-hp-z8
##SBATCH --mail-user=oliviayl@stanford.edu
##SBATCH --mail-type=ALL

cd /iris/u/oliviayl/repos/affordance-learning/vrb
source activate vrb
python demo.py --model_path ./models/model_checkpoint_1249.pth.tar
# python demo.py --image ./kitchen.jpeg --model_path ./models/model_checkpoint_1249.pth.tar
# python demo.py --image /iris/u/oliviayl/repos/affordance-learning/epic_kitchens/DATASETS/EPIC-KITCHENS-100/folder1/P22_114/frame_0000000001.jpg --video P22_114 --model_path ./models/model_checkpoint_1249.pth.tar