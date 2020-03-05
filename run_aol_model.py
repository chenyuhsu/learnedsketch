import os
import sys
import time
import argparse
import random
import datetime
import numpy as np
import tensorflow as tf

from utils.aol_utils import get_data_aol_feat_list
from utils.utils import get_stat, git_log, AverageMeter, keep_latest_files
from utils.nn_utils import fc_layers, write_summary


def construct_graph(args):
    with tf.variable_scope("nhh"):
        feat = tf.placeholder(tf.float32,
                [args.batch_size, args.n_feat], name='feat')
        labels = tf.placeholder(tf.float32, [args.batch_size], name='labels')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        data_len = tf.placeholder(tf.int32, [], name='data_len')
        keep_probs = tf.placeholder(tf.float32, [len(args.keep_probs)], name='keep_probs')

        feat_len  = tf.reshape(feat[:, 0], [args.batch_size, 1])
        feat_char = feat[:, 1:]

        # char-level RNN
        alphabet_size = 44
        # 0 has no meaning
        # a~z -> 1~26, digits -> 27~36,
        # ' ', '.', '-', '\'', '&', ';' -> 42, # 'nv' -> 43
        E = tf.get_variable('embedding_matrix', [alphabet_size, args.embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        embeds = tf.nn.embedding_lookup(E, tf.cast(feat_char, dtype=tf.int32))
        print('embeds', embeds)

        rnn_layers = []
        for hidden_size in args.rnn_hiddens:
            if args.cell == 'LSTM':
                rnn_layers.append(tf.nn.rnn_cell.LSTMCell(hidden_size))
            elif args.cell == 'GRU':
                rnn_layers.append(tf.nn.rnn_cell.GRUCell(hidden_size))
            else:
                assert False

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=embeds, sequence_length=tf.squeeze(feat_len), dtype=tf.float32, scope='char-rnn')
        if args.cell == 'LSTM':
            final_state = rnn_state[-1][0]  # state of the last layer (in the last time step),
                                            # [0] is becuase state for LSTM is a tuple (c, h)
        elif args.cell == 'GRU':
            final_state = rnn_state[-1]

        # fully connected encoder
        feat_mid = tf.concat([feat_len, final_state], axis=1)
        print('feat mid', feat_mid)
        hidden_len = [int(feat_len.shape[1]) + int(final_state.shape[1])] + args.hiddens + [1]

        if args.log_hist:
            enc_sum = range(len(args.hiddens)+1)
        else:
            enc_sum = []

        # important: initialize the layers with differnt random seeds!
        output, weights, bias = fc_layers(feat_mid, hidden_len, keep_probs, name='fc_encoder', activation=args.activation, summary_layers=enc_sum)

        if args.relu_output:
            output = tf.nn.relu(tf.squeeze(output))
        else:
            output = tf.squeeze(output)
        loss = tf.losses.mean_squared_error(labels=labels[:data_len], predictions=output[:data_len])

        # log gradients
        if args.log_hist:
            with tf.name_scope('gradient/summaries'):
                vars_train = tf.trainable_variables()
                grads = tf.gradients(loss, vars_train)
                for grad, var in zip(grads, vars_train):
                    tf.summary.histogram(var.name, grad)
            merged_sum = tf.summary.merge_all()
        else:
            merged_sum = []

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_op = optimizer.minimize(loss)
        print('finished constructing the graph')

    model = {}
    model['feat'] = feat
    model['labels'] = labels
    model['learning_rate'] = learning_rate
    model['update_op'] = update_op
    model['loss'] = loss
    model['merged_sum'] = merged_sum
    model['data_len'] = data_len
    model['keep_probs'] = keep_probs
    model['output'] = output
    return model

def train(model, x, y, args, sess, ite, writer, idx=None):
    if idx is None:
       idx = [i for i in range(len(y))]
       random.shuffle(idx)
    assert len(y) == len(idx), "idx (order) needs to have the same length as y"

    loss_meter = AverageMeter()

    # prepare the indices in advance
    for i in range(0, len(y), args.batch_size):
        batch_idx = idx[i:i+args.batch_size]
        batch_x = x[batch_idx]
        batch_y = y[batch_idx]
        n_flows = len(batch_y)
        if n_flows != args.batch_size:
            batch_x = np.concatenate((batch_x, np.zeros((args.batch_size - n_flows, batch_x.shape[1]))), axis=0)
            batch_y = np.concatenate((batch_y, np.zeros(args.batch_size - n_flows)), axis=0)

        input_feed = {
            model['feat']: batch_x,
            model['labels']: batch_y,
            model['learning_rate']: args.lr,
            model['data_len']: n_flows,
            model['keep_probs']: args.keep_probs,
            }
        _, loss_b = sess.run([model['update_op'], model['loss']], feed_dict=input_feed)
        loss_meter.update(loss_b)

        if ite % 100 == 0:
            write_summary(writer, 'train loss', loss_b, ite)
        ite = ite + 1

    if args.log_hist:
        var_sum = sess.run(model['merged_sum'], feed_dict=input_feed)
        writer.add_summary(var_sum, ite)
    return loss_meter.avg, ite

def evaluate(model, x, y, args, sess, ite=None, writer=None, name=''):
    loss_meter = AverageMeter()
    output_all = np.array([]).reshape((-1, 1))
    for i in range(0, len(y), args.batch_size):
        batch_x = x[i:i+args.batch_size]
        batch_y = y[i:i+args.batch_size]
        n_flows = len(batch_y)
        if n_flows != args.batch_size:
            batch_x = np.concatenate((batch_x, np.zeros((args.batch_size - n_flows, batch_x.shape[1]))), axis=0)
            batch_y = np.concatenate((batch_y, np.zeros(args.batch_size - n_flows)), axis=0)
        input_feed = {
            model['feat']: batch_x,
            model['labels']: batch_y,
            model['learning_rate']: args.lr,
            model['data_len']: n_flows,
            model['keep_probs']: np.ones(len(args.keep_probs)),
            }
        loss_b, output_b = sess.run([model['loss'], model['output']], feed_dict=input_feed)
        loss_meter.update(loss_b)
        output_all = np.concatenate((output_all, output_b[:n_flows].reshape((-1, 1))))

    if writer is not None:
        write_summary(writer, '%s loss' % name, loss_meter.avg, ite)
    return loss_meter.avg, output_all

def run_training(model, train_x, train_y, valid_x, valid_y, test_x, test_y, args, sess, summary_writer):

    n_batch_per_ep = len(train_y) // args.batch_size
    ite = args.start_epoch * n_batch_per_ep + 1
    best_eval_loss = sys.float_info.max

    for ep in range(args.start_epoch, args.n_epochs):
        start_t = time.time()
        train_loss, ite = train(model, train_x, train_y, args, sess, ite, summary_writer)
        train_time = time.time() - start_t

        if ep % args.eval_n_epochs == 0:
            start_t = time.time()
            valid_loss, valid_output = evaluate(model, valid_x, valid_y, args, sess, ite, summary_writer, name='valid')
            test_loss, test_output   = evaluate(model, test_x, test_y, args, sess, ite, summary_writer, name='test')
            eval_time = time.time() - start_t

            # save the best model from validation
            if valid_loss < best_eval_loss:
                best_eval_loss = valid_loss
                file_name = str(args.save_name)+'_'+time_now+'_ep'+str(ep)+'.'+str(args.seed)
                best_saver.save(sess, 'model/'+file_name)

                folder = os.path.join('./predictions/', args.save_name, '') # '' for trailing slash
                if not os.path.exists(folder):
                    os.makedirs(folder)

                np.savez(os.path.join(folder, file_name),
                    args=args,
                    valid_output=valid_output,
                    test_output=test_output,
                    )
                keep_latest_files(folder, n_keep=3)

            res = ("epoch %d, training loss %.4f (%.1f sec), "
                    "valid loss %.4f, test loss %.4f (%.1f sec)") % \
                    (ep, train_loss, train_time, valid_loss, test_loss, eval_time)
        else:
            res = 'epoch %d, training loss %.4f (%.1f sec)' % \
                    (ep, train_loss, train_time)
        print(res)
        fp.write(res+'\n')
        fp.flush()
        summary_writer.flush()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train", type=str, nargs='*', help="training data (.npy file)", default="")
    argparser.add_argument("--valid", type=str, nargs='*', help="validation data (.npy file)", default="")
    argparser.add_argument("--test",  type=str, nargs='*', help="testing data (.npy file)", required=True)
    argparser.add_argument("--save_name", type=str, help="name for the save results", required=True)
    argparser.add_argument("--seed", type=int, help="random seed", default=69)
    argparser.add_argument("--n_epochs", type=int, help="number of epochs for training", default=1)
    argparser.add_argument("--eval_n_epochs", type=int, help="inference on validation and test set every eval_n_epochs epochs", default=20)
    argparser.add_argument("--batch_size", type=int, help="batch size for training", default=128)
    argparser.add_argument('--hiddens', type=int, nargs='*', default=[32], help="# of hidden units for the final layers")
    argparser.add_argument('--rnn_hiddens', type=int, nargs='*', default=[256], help="# of hidden units for the RNN layers")
    argparser.add_argument('--embed_size', type=int, default=64)
    argparser.add_argument('--keep_probs', type=float, nargs='*', default=[], help="dropout probabilities for the final layers")
    argparser.add_argument("--lr", type=float, default = 0.0001, help="learning rate")
    argparser.add_argument("--memory", type=float, default = 1.0, help="GPU memory fraction used for model training")
    argparser.add_argument('--resume_training', type=str, default="", help="Path to a model checkpoint. Use this flag to resume training or run inference.")
    argparser.add_argument('--start_epoch', type=int, default=0, help="For checkpoint and summary logging. Specify this to be the epoch number of the loaded checkpoint +1.")
    argparser.add_argument('--activation', type=str, default="LeakyReLU", help="activation for FC layers")
    argparser.add_argument('--evaluate', action='store_true', default=False, help="Run model evaluation without training.")
    argparser.add_argument('--log_hist', action='store_true', default=False, help="log histogram of gradients during training for debuging")
    argparser.add_argument('--regress_actual', action='store_true', default=False)
    argparser.add_argument("--regress_min", type=float, default=1, help="minimum cutoff for regression")
    argparser.add_argument('--word_max_len', type=int, default=60)
    argparser.add_argument('--cell', type=str, default='LSTM')
    argparser.add_argument('--relu_output', action='store_true', default=False)
    args = argparser.parse_args()

    assert args.train != '' or args.resume_training != '',      "Must provide training data or a model"
    assert not (args.evaluate and not args.resume_training),    "provide a model with --resume"
    assert not (not args.evaluate and (args.train == '' or args.valid == '')), "use --train and --valid for training"
    assert args.batch_size % 2 == 0, "use a multiple of 2 for batch_size"
    assert args.cell in ['LSTM', 'GRU']

    if not args.keep_probs:
        args.keep_probs = np.ones(len(args.hiddens)+1)
    assert len(args.hiddens)+1 == len(args.keep_probs)

    command = ' '.join(sys.argv)
    print(command)
    print(git_log())

    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    required_folders = ['log', 'summary', 'model', 'predictions']
    for folder in required_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    args.n_feat = 1 + args.word_max_len
    start_t = time.time()

    train_x, train_y = get_data_aol_feat_list(args.train)
    valid_x, valid_y = get_data_aol_feat_list(args.valid)
    test_x, test_y = get_data_aol_feat_list(args.test)
    print('Load data time %.1f seconds' % (time.time() - start_t))

    data_stat = get_stat('train before log', train_x, train_y)
    if args.regress_actual:
        rmin = args.regress_min
    else:
        train_y = np.log(train_y)
        valid_y = np.log(valid_y)
        test_y = np.log(test_y)
        rmin = np.log(args.regress_min)

    data_stat += get_stat('train before rmin', train_x, train_y)
    s = 'rmin %.2f, # train_y < min %.2f\n\n' % (rmin, np.sum(train_y < rmin))
    data_stat += s
    print(s)

    train_y[train_y < rmin] = rmin
    valid_y[valid_y < rmin] = rmin
    test_y[test_y < rmin] = rmin

    data_stat += get_stat('train', train_x, train_y)
    data_stat += get_stat('valid', valid_x, valid_y)
    data_stat += get_stat('test', test_x, test_y)

    model = construct_graph(args)
    init = tf.global_variables_initializer()
    best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    # set seeds
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess, open('log/'+str(args.save_name)+'_'+time_now+'_'+str(args.seed)+'.log','w') as fp:
        summary_writer = tf.summary.FileWriter('summary/%s_%s' % (str(args.save_name), str(args.seed)), graph=tf.get_default_graph())
        fp.write(command+'\n')
        fp.write(git_log()+'\n')
        fp.write(data_stat+'\n')

        sess.run(init)

        if args.resume_training != "":
            best_saver.restore(sess, args.resume_training)
            s = 'resume training from %s, start_epoch %d\n' % (args.resume_training, args.start_epoch)
            fp.write(s)
            print(s)

        if not args.evaluate:
            run_training(model, train_x, train_y, valid_x, valid_y,
                    test_x, test_y, args, sess, summary_writer)
        else:
            start_t = time.time()
            train_loss, train_output = evaluate(model, train_x, train_y, args, sess)
            valid_loss, valid_output = evaluate(model, valid_x, valid_y, args, sess)
            test_loss, test_output = evaluate(model, test_x, test_y, args, sess)

            np.savez(args.save_name+'_res',
                train_output=train_output,
                valid_output=valid_output,
                test_output=test_output,
                train_loss=train_loss,
                valid_loss=valid_loss,
                test_loss=test_loss,
                )

            eval_time = time.time() - start_t
            print('evalutation time %.2f sec' % eval_time)

