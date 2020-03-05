import os
import sys
import time
import argparse
import random
import datetime
import numpy as np
import tensorflow as tf

from utils.utils import get_stat, git_log, AverageMeter, keep_latest_files, get_data, get_data_list
from utils.nn_utils import fc_layers, write_summary


def construct_graph(args):
    with tf.variable_scope("nhh"):
        feat = tf.placeholder(tf.float32,
                [args.batch_size, args.n_feat], name='feat')
        labels = tf.placeholder(tf.float32, [args.batch_size], name='labels')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        data_len = tf.placeholder(tf.int32, [], name='data_len')
        keep_probs = tf.placeholder(tf.float32, [len(args.keep_probs)], name='keep_probs')

        n_feat_ip = 32
        feat_src_ip = tf.reshape(feat[:,:n_feat_ip], [args.batch_size, n_feat_ip, 1])
        feat_dst_ip = tf.reshape(feat[:,n_feat_ip:n_feat_ip*2], [args.batch_size, n_feat_ip, 1])

        src_rnn_layers = []
        dst_rnn_layers = []
        for hidden_size in args.rnn_hiddens:
            src_rnn_layers.append(tf.nn.rnn_cell.LSTMCell(hidden_size))
            dst_rnn_layers.append(tf.nn.rnn_cell.LSTMCell(hidden_size))

        src_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(src_rnn_layers)
        dst_multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(dst_rnn_layers)

        src_rnn_outputs, src_rnn_state = tf.nn.dynamic_rnn(cell=src_multi_rnn_cell, inputs=feat_src_ip, dtype=tf.float32, scope='rnn_src_ip')
        dst_rnn_outputs, dst_rnn_state = tf.nn.dynamic_rnn(cell=dst_multi_rnn_cell, inputs=feat_dst_ip, dtype=tf.float32, scope='rnn_dst_ip')

        src_lstm_state = src_rnn_state[-1][0]
        dst_lstm_state = dst_rnn_state[-1][0]

        # src and dst ports
        n_feat_port = 16
        enc1_hidden_len = [n_feat_port] + args.port_hiddens
        enc2_hidden_len = [n_feat_port] + args.port_hiddens

        feat_src_port = feat[:,n_feat_ip*2:n_feat_ip*2+n_feat_port]
        feat_dst_port = feat[:,n_feat_ip*2+n_feat_port:n_feat_ip*2+n_feat_port*2]
        src_port_out, _, _ = fc_layers(feat_src_port, enc1_hidden_len, np.ones(len(enc1_hidden_len)), name='fc_encoder_src_port', activation=args.activation, summary_layers=[])
        dst_port_out, _, _ = fc_layers(feat_dst_port, enc2_hidden_len, np.ones(len(enc2_hidden_len)), name='fc_encoder_dst_port', activation=args.activation, summary_layers=[])

        # protocol
        proto_out = tf.reshape(feat[:,-1], [args.batch_size, 1])

        # combine all features
        output = tf.concat([src_lstm_state, dst_lstm_state, src_port_out, dst_port_out, proto_out], axis=1)
        hidden_len = [args.rnn_hiddens[-1]*2 + args.port_hiddens[-1]*2 + 1] + args.hiddens + [1]
        output, weights, bias = fc_layers(output, hidden_len, keep_probs, name='fc_encoder', activation=args.activation, summary_layers=[])

        output = tf.squeeze(output)
        loss = tf.losses.mean_squared_error(labels=labels[:data_len], predictions=output[:data_len])

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_op = optimizer.minimize(loss)
        print('finished constructing the graph')

    model = {}
    model['feat'] = feat
    model['labels'] = labels
    model['learning_rate'] = learning_rate
    model['update_op'] = update_op
    model['loss'] = loss
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
            test_loss, test_output = evaluate(model, test_x, test_y, args, sess, ite, summary_writer, name='test')
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train", type=str, nargs='*', help="training data (.npy file)", default="")
    argparser.add_argument("--valid", type=str, nargs='*', help="validation data (.npy file)", default="")
    argparser.add_argument("--test", type=str, nargs='*', help="testing data (.npy file)", required=True)
    argparser.add_argument("--save_name", type=str, help="name for the save results", required=True)
    argparser.add_argument("--seed", type=int, help="random seed", default=69)
    argparser.add_argument("--n_examples", type=int, help="# of examples to use per .npy file", default=10000000)
    argparser.add_argument("--n_epochs", type=int, help="number of epochs for training", default=1)
    argparser.add_argument("--eval_n_epochs", type=int, help="inference on validation and test set every eval_n_epochs epochs", default=20)
    argparser.add_argument("--batch_size", type=int, help="batch size for training", default=512)
    argparser.add_argument('--keep_probs', type=float, nargs='*', default=[], help="dropout probabilities for the final layers")
    argparser.add_argument("--lr", type=float, default = 0.0001, help="learning rate")
    argparser.add_argument("--memory", type=float, default = 1.0, help="GPU memory fraction used for model training")
    argparser.add_argument('--resume_training', type=str, default="", help="Path to a model checkpoint. Use this flag to resume training or run inference.")
    argparser.add_argument('--start_epoch', type=int, default=0, help="For checkpoint and summary logging. Specify this to be the epoch number of the loaded checkpoint +1.")
    argparser.add_argument('--activation', type=str, default="LeakyReLU", help="activation for FC layers")
    argparser.add_argument('--evaluate', action='store_true', default=False, help="Run model evaluation without training.")
    argparser.add_argument("--regress_min", type=float, default=1, help="minimum cutoff for regression")
    argparser.add_argument('--rnn_hiddens', type=int, nargs='*', default=[64], help="# of hidden units for the ip RNN layers")
    argparser.add_argument('--port_hiddens', type=int, nargs='*', default=[16, 8], help="# of hidden units for the port FC layers")
    argparser.add_argument('--hiddens', type=int, nargs='*', default=[32, 32], help="# of hidden units for the final layers")
    args = argparser.parse_args()

    assert args.train != '' or args.resume_training != '',      "Must provide training data or a model"
    assert not (args.evaluate and not args.resume_training),    "provide a model with --resume"
    assert not (not args.evaluate and (args.train == '' or args.valid == '')), "use --train and --valid for training"
    assert args.batch_size % 2 == 0, "use a multiple of 2 for batch_size"

    if not args.keep_probs:
        args.keep_probs = np.ones(len(args.hiddens)+1)
    assert len(args.hiddens)+1 == len(args.keep_probs)

    print(' '.join(sys.argv))
    print(git_log())

    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    required_folders = ['log', 'summary', 'model']
    for folder in required_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # load data
    feat_idx = np.arange(11)
    args.n_feat = 8*8+2*16+1    # ip-> 8*8, port-> 2*16, protocol->1
    start_t = time.time()

    train_x, train_y = get_data(args.train, feat_idx, args.n_examples)
    print('train x shape:', train_x.shape, 'y max', np.max(train_y), 'y min', np.min(train_y))

    valid_x, valid_y = get_data(args.valid, feat_idx, args.n_examples)
    print('valid x shape:', valid_x.shape, 'y max', np.max(valid_y), 'y min', np.min(valid_y))

    test_x, test_y = get_data_list(args.test, feat_idx, args.n_examples)
    print('Load data time %.1f seconds' % (time.time() - start_t))
    if not args.evaluate:
        assert len(test_x) == 1, 'test on more than 1 minute (forgot --evaluate?)'

    data_stat = get_stat('train before log', train_x, train_y)
    train_y = np.log(train_y)
    valid_y = np.log(valid_y)
    for i in range(len(test_y)):
        test_y[i] = np.log(test_y[i])
    rmin = np.log(args.regress_min)

    data_stat += get_stat('train before rmin', train_x, train_y)
    s = 'rmin %.2f, # train_y < min %.2f\n\n' % (rmin, np.sum(train_y < rmin))
    data_stat += s
    print(s)

    train_y[train_y < rmin] = rmin
    valid_y[valid_y < rmin] = rmin
    for i in range(len(test_y)):
        test_y[i][test_y[i] < rmin] = rmin

    data_stat += get_stat('train', train_x, train_y)
    data_stat += get_stat('valid', valid_x, valid_y)
    for i in range(len(test_x)):
        data_stat += get_stat('test', test_x[i], test_y[i])

    model = construct_graph(args)
    init = tf.global_variables_initializer()
    best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # set seeds
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess, open('log/'+str(args.save_name)+'_'+time_now+'_'+str(args.seed)+'.log','w') as fp:
        summary_writer = tf.summary.FileWriter('summary/%s_%s' % (str(args.save_name), str(args.seed)), graph=tf.get_default_graph())
        fp.write(' '.join(sys.argv)+'\n')
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
                    test_x[0], test_y[0], args, sess, summary_writer)
        else:
            start_t = time.time()
            train_loss, train_output = evaluate(model, train_x, train_y, args, sess)
            valid_loss, valid_output = evaluate(model, valid_x, valid_y, args, sess)

            test_loss_all = []
            test_output_all = []
            for i in range(len(test_x)):
                test_loss, test_output = evaluate(model, test_x[i], test_y[i], args, sess)
                test_loss_all.append(test_loss)
                test_output_all.append(test_output)
                print('test %d, \ttime %.2f sec' % (i, time.time() - start_t))

            np.savez(args.save_name+'_res',
                train_output=train_output,
                valid_output=valid_output,
                test_output=test_output_all,
                test_loss=test_loss_all,
                )

            eval_time = time.time() - start_t
            print('evalutation time %.2f sec' % eval_time)
