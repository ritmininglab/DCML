
import argparse
from baseline_dcml import main
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='data/CIFAR-100N',#'data/FC100',#'data/miniImageNet',
    help="Directory containing the dataset")
parser.add_argument('--dataset',default='FC100')
parser.add_argument(
    '--model_dir',
    default='experiments/cn_dccp',
    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default=None,
    help="Optional, name of the file in --model_dir containing weights to \
          reload before training")  # 'best' or 'train'
parser.add_argument('--warm_start',default=5000)#5000
parser.add_argument('--gamma',default=0.5)
parser.add_argument('--noise_type',default='sym')
parser.add_argument('--proxy_ckpt_steps',default=2500)#2500
parser.add_argument('--proxy_ckpt_upper_bound',default=30000)#30000
parser.add_argument('--save_summary_steps',default=500)#500
parser.add_argument('--update_curriculum_steps',default=10000)#5000
parser.add_argument('--update_curriculum_steps_E',default=10000)#5000
parser.add_argument('--num_curri_steps',default=100)
parser.add_argument('--class_start_percent',default=0.8)
parser.add_argument('--class_upper_bound',default=0.95)
parser.add_argument('--E-CL',default=True)
parser.add_argument('--growing_factor',default=1.5)
parser.add_argument('--threshold_e',default=0.2)#0.8
parser.add_argument('--class_select_num',default=30)
parser.add_argument('--remove_class_num_upperb',default=10)
parser.add_argument('--train_noise_class_step',default=5)
parser.add_argument('--class_metrics',default='cp',help='cp, cp_asy,loss')
parser.add_argument('--CMAML',default=True)
# parser.add_argument('--num_episodes',default=100)
# parser.add_argument('--num_classes',default=5)
# parser.add_argument('--num_samples',default=1)
# parser.add_argument('--num_query',default=15)
# parser.add_argument('--num_steps',default=10)
# parser.add_argument('--num_inner_tasks',default=8)
# parser.add_argument('--num_train_updates',default=1)
# parser.add_argument('--num_eval_updates',default=3)
# parser.add_argument('--num_workers',default=1)
# parser.add_argument('--SEED',default=1)
# parser.add_argument('--meta_lr',default=1e-3)
# parser.add_argument('--task_lr',default=1e-1)



args=parser.parse_args()
main(args=parser.parse_args())
