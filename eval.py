import argparse
import time
import logging
import os,sys
import mxnet as mx

def score(data_path, metrics, gpus, batch_size,
          image_shape='3,224,224', data_nthreads=4, 
          label_name='softmax_label', max_num_examples=None,
          model_path='./'):
    # create data iterator
    data_mean = [123.68,116.78,103.94]
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    data = mx.io.ImageRecordIter(
        path_imgrec=data_path,
        label_width=1,
        data_name=b'data',
        label_name=b'softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False,
        round_batch=False,
        mean_r=data_mean[0],
        mean_g=data_mean[1],
        mean_b=data_mean[2],
        scale=0.017,
        num_parts=1,
        part_index=0)

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 1)

    # create module
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name,])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num > max_num_examples:
            break
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_shape', type=str, default='3,224,224')
    parser.add_argument('--data_nthreads', type=int, default=4,
                        help='number of threads for data decoding')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(data_path=args.data_path,
                    metrics = metrics,
                    gpus=args.gpus,
                    batch_size=args.batch_size,
                    model_path=args.model_path)
    logging.info('Finished with %f images per second', speed)

    for m in metrics:
        logging.info(m.get())
