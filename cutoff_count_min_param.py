import os
import sys
import time
import argparse
import numpy as np
from itertools import repeat

from multiprocessing import Pool
from utils.utils import get_stat, git_log, get_data_str_with_ports_list
from utils.aol_utils import get_data_aol_query_list
from sketch import cutoff_countmin, cutoff_lookup, cutoff_countmin_wscore, order_y_wkey_list
from sketch import cutoff_countsketch, cutoff_countsketch_wscore


def run_ccm(y, b_cutoff, n_hashes, n_buckets, name, sketch_type):
    start_t = time.time()
    if sketch_type == 'count_min':
        loss, space = cutoff_countmin(y, n_buckets, b_cutoff, n_hashes)
    else:
        loss, space = cutoff_countsketch(y, n_buckets, b_cutoff, n_hashes)
    print('%s: bcut: %d, # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, b_cutoff, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss, space

def run_ccm_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes, name, sketch_type):
    start_t = time.time()
    if sketch_type == 'count_min':
        loss, space = cutoff_countmin_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes)
    else:
        loss, space = cutoff_countsketch_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes)
    print('%s: scut: %.3f, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, score_cutoff, n_hashes, n_cm_buckets, loss, time.time() - start_t))
    return loss, space

def run_ccm_lookup(x, y, n_hashes, n_cm_buckets, d_lookup, s_cutoff, name, sketch_type):
    start_t = time.time()
    if sketch_type == 'count_min':
        loss, space = cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, s_cutoff)
    else:
        loss, space = cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, s_cutoff, \
            sketch='CountSketch')
    print('%s: s_cut: %d, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, s_cutoff, n_hashes, n_cm_buckets, loss, time.time() - start_t))
    return loss, space

def get_great_cut(b_cut, y, max_bcut):
    assert b_cut <= max_bcut
    y_sorted = np.sort(y)[::-1]
    if b_cut < len(y_sorted):
        s_cut = y_sorted[b_cut]
    else:
        s_cut = y_sorted[-1]

    # return cut at the boundary of two frequencies
    n_after_same = np.argmax((y_sorted == s_cut)[::-1]) # items after items == s_cut
    if (len(y) - n_after_same) < max_bcut:
        b_cut_new = (len(y) - n_after_same)
        if n_after_same == 0:
            s_cut = s_cut - 1   # get every thing
        else:
            s_cut = y_sorted[b_cut_new] # item right after items == s_cut
    else:
        b_cut_new = np.argmax(y_sorted == s_cut) # first item that # items == s_cut
    return b_cut_new, s_cut


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_results", type=str, nargs='*', help="testing results of a model (.npz file)", default='')
    argparser.add_argument("--valid_results", type=str, nargs='*', help="validation results of a model (.npz file)", default='')
    argparser.add_argument("--test_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--valid_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--lookup_data", type=str, nargs='*', help="list of input .npy data", default=[])
    argparser.add_argument("--save", type=str, help="prefix to save the results", required=True)
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", required=True)
    argparser.add_argument("--perfect_order", action='store_true', default=False)
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=10)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--count_sketch", action='store_true', default=False)
    args = argparser.parse_args()

    assert not (args.perfect_order and args.lookup_data),   "use either --perfect or --lookup"

    command = ' '.join(sys.argv) + '\n'
    log_str = command
    log_str += git_log() + '\n'
    print(log_str)
    np.random.seed(args.seed)

    if args.count_sketch:
        sketch_type = 'count_sketch'
    else:
        sketch_type = 'count_min'

    if args.perfect_order:
        name = 'cutoff_%s_param_perfect' % sketch_type
    elif args.lookup_data:
        name = 'lookup_table_%s' % sketch_type
    else:
        name = 'cutoff_%s_param' % sketch_type

    folder = os.path.join('param_results', name, '')
    if not os.path.exists(folder):
        os.makedirs(folder)

    start_t = time.time()
    if args.aol_data:
        x_valid, y_valid = get_data_aol_query_list(args.valid_data)
        x_test, y_test = get_data_aol_query_list(args.test_data)
    else:
        x_valid, y_valid = get_data_str_with_ports_list(args.valid_data)
        x_test, y_test = get_data_str_with_ports_list(args.test_data)
    log_str += get_stat('valid data:\n'+'\n'.join(args.valid_data), x_valid, y_valid)
    log_str += get_stat('test data:\n'+'\n'.join(args.test_data), x_test, y_test)

    if args.lookup_data:
        if args.aol_data:
            x_train, y_train = get_data_aol_query_list(args.lookup_data)
        else:
            x_train, y_train = get_data_str_with_ports_list(args.lookup_data)
        log_str += get_stat('lookup data:\n'+'\n'.join(args.lookup_data), x_train, y_train)
    print('data loading time: %.1f sec' % (time.time() - start_t))

    if args.valid_results:
        key = 'valid_output'
        y_valid_ordered, valid_scores = order_y_wkey_list(y_valid, args.valid_results, key)

    if args.test_results:
        key = 'test_output'
        y_test_ordered, test_scores = order_y_wkey_list(y_test, args.test_results, key)

    if args.perfect_order:
        assert np.abs(1 - len(y_valid) / len(y_test)) < 0.1,   "valid and test data should have similar # items"

    cutoff_cost_mul = 2 # cutoff buckets cost x2
    bcut_all = []
    scut_all = []
    nh_all = []
    nb_all = []
    for space in args.space_list:
        max_bcut = space * 1e6 / (4 * cutoff_cost_mul)
        b_cutoffs = np.linspace(0.1, 0.9, 9) * max_bcut
        for bcut in b_cutoffs:
            for n_hash in args.n_hashes_list:
                bcut = int(bcut)
                if args.perfect_order:
                    scut = 0    # version 2, scut is not used
                elif args.lookup_data:
                    bcut, scut = get_great_cut(bcut, y_train, np.floor(max_bcut))    # this has to be y_train
                else:
                    if bcut < len(y_valid):
                        scut = valid_scores[bcut]
                    else:
                        scut = valid_scores[-1]
                n_cmin_buckets = int((space * 1e6 - bcut * 4 * cutoff_cost_mul) / (n_hash * 4))
                bcut_all.append(bcut)
                scut_all.append(scut)
                nh_all.append(n_hash)
                nb_all.append(bcut + n_cmin_buckets)
    rshape = (len(args.space_list), len(b_cutoffs), len(args.n_hashes_list))
    n_cm_all = np.array(nb_all) - np.array(bcut_all)

    if args.lookup_data:
        min_scut = np.min(scut_all) # no need to store elements that are smaller
        x_train = np.asarray(x_train)
        x_train_hh = x_train[y_train > min_scut]
        y_train_hh = y_train[y_train > min_scut]
        lookup_dict = dict(zip(x_train_hh, y_train_hh))

    start_t = time.time()
    pool = Pool(args.n_workers)
    if args.perfect_order:
        y_sorted = np.sort(y_valid)[::-1]
        results = pool.starmap(run_ccm, zip(repeat(y_sorted), bcut_all, nh_all, nb_all, repeat(name), repeat(sketch_type)))
    elif args.lookup_data:
        results = pool.starmap(run_ccm_lookup, zip(repeat(x_valid), repeat(y_valid), nh_all, n_cm_all, repeat(lookup_dict), scut_all, repeat(name), repeat(sketch_type)))
    else:
        results = pool.starmap(run_ccm, zip(repeat(y_valid_ordered), bcut_all, nh_all, nb_all, repeat(name), repeat(sketch_type)))
    pool.close()
    pool.join()
    valid_results, space_actual = zip(*results)
    valid_results = np.reshape(valid_results, rshape)
    space_actual = np.reshape(space_actual, rshape)
    bcut_all = np.reshape(bcut_all, rshape)
    scut_all = np.reshape(scut_all, rshape)
    nh_all = np.reshape(nh_all, rshape)
    nb_all = np.reshape(nb_all, rshape)

    log_str += '==== valid_results ====\n'
    for i in range(len(valid_results)):
        log_str += 'space: %.2f\n' % args.space_list[i]
        for j in range(len(valid_results[i])):
            for k in range(len(valid_results[i, j])):
                log_str += '%s: bcut: %d, # hashes %d, # buckets %d - \tloss %.2f\tspace %.1f\n' % \
                    (name, bcut_all[i,j,k], nh_all[i,j,k], nb_all[i,j,k], valid_results[i,j,k], space_actual[i,j,k])
    log_str += 'param search done -- time: %.2f sec\n' % (time.time() - start_t)

    np.savez(os.path.join(folder, args.save+'_valid'),
        command=command,
        loss_all=valid_results,
        b_cutoffs=bcut_all,
        n_hashes=nh_all,
        n_buckets=nb_all,
        space_list=args.space_list,
        space_actual=space_actual,
        )

    log_str += '==== best parameters ====\n'
    rshape = (len(args.space_list), -1)
    best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    best_scuts     = scut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_bcuts     = bcut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    best_valid_space = space_actual.reshape(rshape)[np.arange(rshape[0]), best_param_idx]

    for i in range(len(best_valid_loss)):
        log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
            (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], best_valid_loss[i], best_valid_space[i])

    # test data using best parameters
    pool = Pool(args.n_workers)
    if args.perfect_order:
        # version 2
        y_sorted = np.sort(y_test)[::-1]
        results = pool.starmap(run_ccm, zip(repeat(y_sorted), best_bcuts, best_n_hashes, best_n_buckets, repeat(name), repeat(sketch_type)))
    elif args.lookup_data:
        results = pool.starmap(run_ccm_lookup,
            zip(repeat(x_test), repeat(y_test), best_n_hashes, best_n_buckets - best_bcuts, repeat(lookup_dict), best_scuts, repeat(name), repeat(sketch_type)))
    else:
        results = pool.starmap(run_ccm_wscore,
            zip(repeat(y_test_ordered), repeat(test_scores), best_scuts, best_n_buckets - best_bcuts, best_n_hashes, repeat(name), repeat(sketch_type)))
    pool.close()
    pool.join()

    test_results, space_test = zip(*results)

    log_str += '==== test test_results ====\n'
    for i in range(len(test_results)):
        log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
               (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], test_results[i], space_test[i])

    log_str += 'total time: %.2f sec\n' % (time.time() - start_t)
    print(log_str)
    with open(os.path.join(folder, args.save+'.log'), 'w') as f:
        f.write(log_str)

    np.savez(os.path.join(folder, args.save+'_test'),
        command=command,
        loss_all=test_results,
        s_cutoffs=best_scuts,
        b_cutoffs=best_bcuts,
        n_hashes=best_n_hashes,
        n_buckets=best_n_buckets,
        space_list=args.space_list,
        space_actual=space_test,
        )

