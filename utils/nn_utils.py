import tensorflow as tf


def fc_layers(input, hidden_len, hidden_keeps, name='fc_layers', activation='LeakyReLU', summary_layers=[0, 4, 9], seed=None):
    with tf.variable_scope(name):
        weight = []
        bias = []
        output = input
        for i in range(len(hidden_len) - 1):
            w = tf.get_variable('w%d' % i, shape=[hidden_len[i], hidden_len[i+1]],
                    initializer=tf.truncated_normal_initializer(stddev=0.2, seed=seed))
            b = tf.get_variable('bias%d' % i, shape=[hidden_len[i+1]],
                    initializer=tf.zeros_initializer())
            if i in summary_layers:
                variable_summaries(w, name='w%d' % i)
                variable_summaries(b, name='bias%d' % i)
            weight.append(w)
            bias.append(b)
            output = tf.nn.dropout(output, keep_prob = hidden_keeps[i])
            output = tf.matmul(output, weight[-1]) + bias[-1]
            print('hidden %d:' % i, output)
            if i != len(hidden_len) - 2:
                if activation == 'sigmoid':
                    output = tf.nn.sigmoid(output)
                elif activation == 'LeakyReLU':
                    output = tf.nn.leaky_relu(output)
                else:
                    assert False, print('activation %s not supported' % activation)

        print('fc layers %s done' % name)
    return output, weight, bias


def write_summary(writer, name, val, ite):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=val)
    writer.add_summary(summary, ite)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name+'/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
