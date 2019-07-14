import os
import argparse
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    data_loader = get_loader(config.image_path, config.image_size, config.dataset,
                             config.batch_size, config.num_workers)

    # Solver
    solver = Solver(data_loader, config)

    # Create directories if not exist
    if not os.path.exists(os.path.join(config.log_path, config.version)):
        os.makedirs(os.path.join(config.log_path, config.version))
    if not os.path.exists(os.path.join(config.model_save_path, config.version)):
        os.makedirs(os.path.join(config.model_save_path, config.version))
    if not os.path.exists(os.path.join(config.sample_path, config.version)):
        os.makedirs(os.path.join(config.sample_path, config.version))
    if not os.path.exists(os.path.join(config.result_path, config.version)):
        os.makedirs(os.path.join(config.result_path, config.version))

    if config.mode == 'train':
        solver.train_gan_with_csvdd()
    elif config.mode == 'test':
        solver.test_plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=2)
    parser.add_argument('--image_channel', type=int, default=1)
    parser.add_argument('--num_gen', type=int, default=1)
    parser.add_argument('--num_dis', type=int, default=1)
    parser.add_argument('--num_ocsvm', type=int, default=8)
    parser.add_argument('--ms_num_image', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_f_dim', type=int, default=128)
    parser.add_argument('--d_f_dim', type=int, default=128)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_ncl', type=float, default=0.1)
    parser.add_argument('--lambda_s', type=float, default=1)
    parser.add_argument('--softmax_beta', type=float, default=10)
    parser.add_argument('--lambda_d', type=float, default=1)
    parser.add_argument('--n_viz', type=int, default=5120)
    parser.add_argument('--mog_scale', type=float, default=2.0)
    # Training setting
    parser.add_argument('--total_step', type=int, default=25000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gum_orig', type=float, default=1)  # gum start temperature
    parser.add_argument('--gum_temp', type=float, default=1)
    parser.add_argument('--min_temp', type=float, default=0.01)
    parser.add_argument('--gum_temp_decay', type=float, default=0.0001)
    parser.add_argument('--step_t_decay', type=int, default=1)  # epoch to apply decaying
    parser.add_argument('--pretrained_model', type=int, default=None)
    parser.add_argument('--start_anneal', type=int, default=0)  # epoch to start annealing
    parser.add_argument('--run', type=int, default=1)
 
    # Test setting
    parser.add_argument('--test_size', type=int, default=64)
    parser.add_argument('--test_model', type=str, default='50000_G.pth')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--test_ver', type=str, default='tsne')
    parser.add_argument('--version', type=str, default='sr-gan_gauss')
    parser.add_argument('--nrow', type=int, default=8)
    parser.add_argument('--ncol', type=int, default=8)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='ring', choices=['ring', 'grid'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=8)
    parser.add_argument('--num_cluster', type=int, default=8)

    # Load balance
    parser.add_argument('--load_balance', type=str2bool, default=False)
    parser.add_argument('--balance_weight', type=float, default=1.0)
    parser.add_argument('--matching_weight', type=float, default=1.0)  # for 2, for 5 1000, for 4500

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=780)
    parser.add_argument('--score_epoch', type=int, default=3)  # = 5 epochs
    parser.add_argument('--score_start', type=int, default=3)  # start at 5 (default)

    config = parser.parse_args()
    #print(config)
    main(config)
