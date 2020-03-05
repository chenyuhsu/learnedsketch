import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def get_best_loss_space(data):
    rshape = (len(data['space_list']), -1)
    best_param_idx = np.argmin(data['loss_all'].reshape(rshape), axis=1)
    loss = data['loss_all'].reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    space_actual = data['space_actual'].reshape(rshape)[np.arange(rshape[0]), best_param_idx] / 1e6
    return loss, space_actual


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--count_min", type=str, default='')
    argparser.add_argument("--learned_cmin", type=str, nargs='*', default=[])
    argparser.add_argument("--model_names", type=str, nargs='*', default=["Learned Count Min"])
    argparser.add_argument("--perfect_ccm", type=str, default='')
    argparser.add_argument("--lookup_table_ccm", type=str, default='')
    argparser.add_argument("--model_sizes", type=float, nargs='*', default=[])
    argparser.add_argument("--lookup_size", type=float, default=0.0)
    argparser.add_argument("--x_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--y_lim", type=float, nargs='*', default=[])
    argparser.add_argument("--title", type=str, default='')
    argparser.add_argument("--algo", type=str, default='Alg')
    args = argparser.parse_args()

    if args.learned_cmin:
        if not args.model_sizes:
            args.model_sizes = np.zeros(len(args.learned_cmin))
        assert len(args.learned_cmin) == len(args.model_names), "provide names for the learned_cmin results"
        assert len(args.learned_cmin) == len(args.model_sizes), "provide model sizes for the learned_cmin results"

    ax = plt.figure().gca()

    if args.count_min:
        data = np.load(args.count_min)
        space_cmin = data['space_list']
        loss_cmin = np.amin(data['loss_all'], axis=0)
        ax.plot(space_cmin, loss_cmin, label=args.algo)

    if args.lookup_table_ccm:
        data = np.load(args.lookup_table_ccm)
        if len(data['loss_all'].shape) == 1:
            print('plot testing results for lookup table')
            ax.plot(data['space_actual'] / 1e6 + args.lookup_size, data['loss_all'], linestyle='-.', label='Table lookup ' + args.algo)
        else:
            loss_lookup, space_actual = get_best_loss_space(data)
            ax.plot(space_actual, loss_lookup + args.lookup_size, linestyle='-.', label='Table lookup ' + args.algo)

    if args.perfect_ccm:
        data = np.load(args.perfect_ccm)
        if len(data['loss_all'].shape) == 1:
            print('plot testing results for perfect CCM')
            ax.plot(data['space_actual'] / 1e6, data['loss_all'], linestyle='-.', label='Learned ' + args.algo + ' (Ideal)')
        else:
            loss_cutoff_pf, space_actual = get_best_loss_space(data)
            ax.plot(space_cmin, loss_cutoff_pf, linestyle='--', label='Learned ' + args.algo + ' (Ideal)')

    if args.learned_cmin:
        for i, cmin_result in enumerate(args.learned_cmin):
            data = np.load(cmin_result)
            if len(data['loss_all'].shape) == 1:
                print('plot testing results for cutoff cmin')
                ax.plot(data['space_actual'] / 1e6 + args.model_sizes[i], data['loss_all'], label=args.model_names[i])
            else:
                loss_cutoff, space_actual = get_best_loss_space(data)
                ax.plot(space_actual + args.model_sizes[i], loss_cutoff, label=args.model_names[i])

    ax.set_ylabel('loss')
    ax.set_xlabel('space (MB)')
    if args.y_lim:
        ax.set_ylim(args.y_lim)
    if args.x_lim:
        ax.set_xlim(args.x_lim)

    title = 'loss vs space - %s' % args.title
    ax.set_title(title)
    plt.legend(loc="upper right")
    plt.show()

