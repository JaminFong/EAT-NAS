import mxnet as mx
import logging

def get_setting_params(**kwargs):
    # bn_params
    bn_mom = kwargs.get('bn_mom', 0.9)
    bn_eps = kwargs.get('bn_eps', 2e-5)
    fix_gamma = kwargs.get('fix_gamma', False)
    use_global_stats = kwargs.get('use_global_stats', False)

    # net_setting param
    workspace = kwargs.get('workspace', 512)
    group_base = kwargs.get('group_base', 1)

    setting_params={}
    setting_params['bn_mom']=bn_mom
    setting_params['bn_eps']=bn_eps
    setting_params['fix_gamma'] = fix_gamma
    setting_params['use_global_stats'] = use_global_stats
    setting_params['workspace']=workspace
    setting_params['group_base'] =group_base

    return setting_params

'''
ConvOP [0,1,2] sep-conv/0, mobile-ib-conv-3/1, mobile-ib-conv-6/2.
KernelSize [0,1,2] 3x3/0, 5x5/1, 7x7/2.
SkipOp [0,1] no/0, id/1.
Layers [0,1,2,3] 1/0, 2/1, 3/2, 4/3.
WidthFactor [0.5, 1.0, 1.5, 2.0]
'''

def get_eatnet_param(net_code,_mbv2base=True):
    assert type(net_code) is list
    _basic_chs = [16, 24, 32, 64, 96, 160, 320]
    block_chs = [32, ]

    input_output_filter = []
    conv_ops = []
    repeat_num = []
    first_stride = [0, 1, 1, 1, 0, 1, 0]
    is_id_skip = []
    kernel_size = []
    filter_depth = []
    for i in range(len(net_code)):
        conv_ops.append('sep-conv' if net_code[i][0]<1 else 'mb-conv')
        repeat_num.append(net_code[i][3]+1)
        is_id_skip.append(net_code[i][2]>0)
        kernel_size.append((net_code[i][1]+1)*2+1)
        # filter_depth
        if net_code[i][0]<1:
            filter_depth.append(-1)
        elif net_code[i][0]==1:
            filter_depth.append(3)
        elif net_code[i][0]==2:
            filter_depth.append(6)
        else:
            raise ValueError('Wrong conv_ops {}'.format(conv_ops))
        # input_output_filter
        if _mbv2base == True:
            block_chs.append(int(_basic_chs[i] * net_code[i][-1]))
        else:
            block_chs.append(int(block_chs[i] * net_code[i][-1]))

        input_output_filter.append([block_chs[-2],block_chs[-1]])

    num_stage = 7
    net_params={}
    net_params['conv_ops']=conv_ops
    net_params['num_stage']=num_stage
    net_params['repeat_num']=repeat_num
    net_params['input_output_filter']=input_output_filter
    net_params['first_stride']=first_stride
    net_params['is_id_skip'] = is_id_skip
    net_params['kernel_size']=kernel_size
    net_params['filter_depth'] = filter_depth

    return net_params


def inverted_residual_block(data,
                            input_channels,
                            output_channels,
                            setting_params,
                            multiplier=1,
                            kernel=(3,3),
                            stride=(1,1),
                            t=4,
                            id_skip=True,
                            dilate=1,
                            with_dilate=False,
                            name=None,
                            *args,
                            **kwargs):

    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    group_base = setting_params['group_base']

    assert stride[0] == stride[1]
    in_channels= int(input_channels*multiplier)*t
    out_channels=int(output_channels*multiplier)
    pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)

    if id_skip:
        if (input_channels == output_channels) and (stride[0]==1):
            short_cut = data
        else:
            bottleneck_bypass = mx.sym.Convolution(data=data,
                                                   num_filter=out_channels,
                                                   kernel=(1, 1),
                                                   pad=(0, 0),
                                                   stride=(1, 1) if with_dilate else stride,
                                                   no_bias=True,
                                                   num_group=1,
                                                   workspace=workspace,
                                                   name=name + '_bypass_conv')

            bottleneck_bypass = mx.sym.BatchNorm(data=bottleneck_bypass,
                                                 fix_gamma=fix_gamma,
                                                 eps=bn_eps,
                                                 momentum=bn_mom,
                                                 use_global_stats=use_global_stats,
                                                 name=name + '_bypass_bn')
            short_cut= bottleneck_bypass

    if with_dilate:
        stride=(1, 1)

    bottleneck_a = mx.sym.Convolution(data=data,
                                      num_filter=in_channels,
                                      kernel=(1,1),
                                      pad=(0,0),
                                      stride=(1,1),
                                      no_bias=True,
                                      num_group=1,
                                      workspace=workspace,
                                      name=name + '_conv2d_pointwise')

    bottleneck_a = mx.sym.BatchNorm(data=bottleneck_a,
                                    fix_gamma=fix_gamma,
                                    eps=bn_eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name + '_conv2d_pointwise_bn')

    bottleneck_a = mx.sym.Activation(data=bottleneck_a,
                                     act_type='relu',
                                     name=name + '_conv2d_pointwise_relu')

    bottleneck_b = mx.sym.Convolution(data=bottleneck_a,
                                      num_filter=in_channels,
                                      kernel=kernel,
                                      pad=pad,
                                      stride=stride,
                                      no_bias=True,
                                      num_group=int(in_channels/group_base),
                                      dilate=(dilate, dilate),
                                      workspace=workspace,
                                      name=name + '_conv2d_depthwise')

    bottleneck_b = mx.sym.BatchNorm(data=bottleneck_b,
                                    fix_gamma=fix_gamma,
                                    eps=bn_eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name + '_conv2d_depthwise_bn')

    bottleneck_b = mx.sym.Activation(data=bottleneck_b,
                                     act_type='relu',
                                     name=name + '_conv2d_depthwise_relu')


    bottleneck_c = mx.sym.Convolution(data=bottleneck_b,
                                      num_filter=out_channels,
                                      kernel=(1, 1),
                                      pad=(0, 0),
                                      stride=(1, 1),
                                      no_bias=True,
                                      num_group=1,
                                      workspace=workspace,
                                      name=name + '_conv2d_linear_transform')

    bottleneck_c = mx.sym.BatchNorm(data=bottleneck_c,
                                    fix_gamma=fix_gamma,
                                    eps=bn_eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name + '_conv2d_linear_transform_bn')
    if id_skip:
        out_data=bottleneck_c+short_cut
    else:
        out_data=bottleneck_c
    return out_data


def separable_conv2d(data,
                     input_channels,
                     output_channels,
                     setting_params,
                     kernel,
                     id_skip=False,
                     stride=(1,1),
                     bias=False,
                     bn_dw_out=True,
                     act_dw_out=True,
                     bn_pw_out=True,
                     act_pw_out=True,
                     dilate=1,
                     with_dilate=False,
                     name=None,
                     *args,
                     **kwargs
                     ):
    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    group_base = setting_params['group_base']
    pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)

    if id_skip:
        if (input_channels == output_channels) and (stride[0]==1):
            short_cut = data
        else:
            bottleneck_bypass = mx.sym.Convolution(data=data,
                                                   num_filter=output_channels,
                                                   kernel=(1, 1),
                                                   pad=(0, 0),
                                                   stride=(1, 1) if with_dilate else stride,
                                                   no_bias=True,
                                                   num_group=1,
                                                   workspace=workspace,
                                                   name=name + '_bypass_conv')

            bottleneck_bypass = mx.sym.BatchNorm(data=bottleneck_bypass,
                                                 fix_gamma=fix_gamma,
                                                 eps=bn_eps,
                                                 momentum=bn_mom,
                                                 use_global_stats=use_global_stats,
                                                 name=name + '_bypass_bn')
            short_cut= bottleneck_bypass

    if with_dilate:
        stride = (1, 1)
    #depthwise
    dw_out = mx.sym.Convolution(data=data,
                                num_filter=input_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                no_bias=False if bias else True,
                                num_group=int(input_channels/group_base),
                                dilate=(dilate,dilate),
                                workspace=workspace,
                                name=name +'_conv2d_depthwise')
    if bn_dw_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma,
                                  eps=bn_eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  name=name+'_conv2d_depthwise_bn')
    if act_dw_out:
        dw_out = mx.sym.Activation(data=dw_out,
                                   act_type='relu',
                                   name=name+'_conv2d_depthwise_relu')
    #pointwise
    pw_out = mx.sym.Convolution(data=dw_out,
                                num_filter=output_channels,
                                kernel=(1, 1),
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True,
                                workspace=workspace,
                                name=name+'_conv2d_pointwise')
    if bn_pw_out:

        pw_out = mx.sym.BatchNorm(data=pw_out,
                                  fix_gamma=fix_gamma,
                                  eps=bn_eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  name=name + '_conv2d_pointwise_bn')
    if act_pw_out:

        pw_out = mx.sym.Activation(data=pw_out,
                                   act_type='relu',
                                   name=name + '_conv2d_pointwise_relu')

    if id_skip:
        out_data=pw_out#+short_cut
    else:
        out_data=pw_out
    return out_data


def add_stage_backbone_block(data,
                             conv_ops,
                             setting_params,
                             stage,
                             repeat_num,
                             input_output_filter,
                             first_stride,
                             is_id_skip,
                             kernel_size,
                             filter_depth,
                             multiplier=1,
                             dilate=1,
                             with_dilate=False,
                             name=None):
    conv = inverted_residual_block if conv_ops[stage]=='mb-conv' else separable_conv2d

    data =conv(data=data,
               setting_params=setting_params,
               input_channels=input_output_filter[stage][0],
               output_channels=input_output_filter[stage][1],
               kernel=(kernel_size[stage], kernel_size[stage]),
               stride=(2, 2) if first_stride[stage] else (1, 1),
               t=filter_depth[stage],
               id_skip=True if is_id_skip[stage] else False,
               multiplier=multiplier,
               dilate=dilate,
               with_dilate=with_dilate,
               name=name+'_stage%d_unit%d_%s' % (stage + 1, 1, conv_ops[stage])
               if name else 'stage%d_unit%d_%s' % (stage + 1, 1, conv_ops[stage]))

    for j in range(repeat_num[stage] - 1):
        data =conv(data=data,
                   setting_params=setting_params,
                   input_channels=input_output_filter[stage][1],
                   output_channels=input_output_filter[stage][1],
                   kernel=(kernel_size[stage], kernel_size[stage]),
                   stride=(1, 1),
                   t=filter_depth[stage],
                   id_skip=True if is_id_skip[stage] else False,
                   multiplier=multiplier,
                   dilate=dilate,
                   with_dilate=with_dilate,
                   name=name+'_stage%d_unit%d_%s' % (stage + 1, j + 2, conv_ops[stage])
                   if name else 'stage%d_unit%d_%s' % (stage + 1, j + 2, conv_ops[stage]))

    return data, data


def add_head_block(data,
                   num_filter,
                   setting_params,
                   multiplier,
                   kernel=(3, 3),
                   stride=(2, 2),
                   pad=(1, 1),
                   name=None):

    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    channels = int(num_filter * multiplier)
    conv1 = mx.sym.Convolution(data=data,
                               num_filter=channels,
                               kernel=kernel,
                               pad=pad,
                               stride=stride,
                               no_bias=True,
                               num_group=1,
                               workspace=workspace,
                               name=name+'_conv1'if name else 'conv1')
    conv1 = mx.sym.BatchNorm(data=conv1,
                             fix_gamma=fix_gamma,
                             eps=bn_eps,
                             momentum=bn_mom,
                             use_global_stats=use_global_stats,
                             name=name+'_conv1_bn' if name else 'conv1_bn')

    conv1 = mx.sym.Activation(data=conv1,
                              act_type='relu',
                              name=name+'_conv1_relu' if name else 'conv1_relu')
    return conv1

def add_conv_1x1(data,
                 num_filter,
                 setting_params,
                 name=None):

    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']

    data = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=(1, 1),
                              stride=(1, 1),
                              pad=(0, 0),
                              no_bias=True,
                              workspace=workspace,
                              name=name+'_stage_conv1x1' if name else 'stage_conv1x1')

    data = mx.sym.BatchNorm(data=data,
                            fix_gamma=fix_gamma,
                            eps=bn_eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            name=name+'_stage_conv1x1_bn' if name else 'stage_conv1x1_bn')

    data = mx.sym.Activation(data=data,
                             act_type='relu',
                             name=name+'_stage_conv1x1_relu' if name else 'stage_conv1x1_relu')
    return data

def add_fc_cls_block(data,
                    grad_scale=1,
                    label_smooth=False,
                    smooth_alpha=0.1,
                    num_class=1000,
                    is_training=True,
                    softmax_name=None,
                    name=None):
    pool1 = mx.symbol.Pooling(data=data,
                              global_pool=True,
                              pool_type='avg',
                              name=name+'_global_avg_pooling' if name else 'global_avg_pooling')
    flat = mx.symbol.Flatten(data=pool1)
    fc = mx.symbol.FullyConnected(data=flat,
                                  num_hidden=num_class,
                                  name=name+'_fc1' if name else 'fc1')

    softmax = mx.symbol.SoftmaxOutput(data=fc,
                                    grad_scale=grad_scale,
                                    smooth_alpha=smooth_alpha if label_smooth else 0.0,
                                    name=softmax_name if softmax_name else 'softmax')
    return softmax


def conv_bn_relu(data, num_filter, kernel, setting_params, pad=(0,0), stride=(1,1), num_group=1, name=None):
    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']

    data = mx.sym.Convolution(data=data,
                            num_filter=num_filter,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            no_bias=True,
                            num_group=num_group,
                            workspace=workspace,
                            name=name + '_conv2d')

    data = mx.sym.BatchNorm(data=data,
                            fix_gamma=fix_gamma,
                            eps=bn_eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            name=name + '_bn')

    data = mx.sym.Activation(data=data,
                            act_type='relu',
                            name=name + '_relu')
    return data


def auxiliary_head(data, setting_params, aux_head_weight=0.4, num_class=1000, name='aux_head'):
    data1 = mx.sym.Activation(data=data, act_type='relu', name=name + '_relu')
    data = mx.symbol.Pooling(data=data,
                            kernel=(5, 5),
                            pool_type='avg',
                            global_pool=False,
                            stride=(2,2),
                            name=name+'_avg_pooling')
    data = conv_bn_relu(data=data1, 
                        num_filter=128, 
                        kernel=(1,1), 
                        setting_params=setting_params, 
                        name=name+'_conv1')
    data = conv_bn_relu(data=data, 
                        num_filter=768, 
                        kernel=(2,2), 
                        setting_params=setting_params, 
                        name=name+'_conv2')
    data = add_fc_cls_block(data=data, 
                            grad_scale=aux_head_weight, 
                            num_class=num_class, 
                            softmax_name='aux_softmax',
                            name=name)
    return data


def get_symbol(**kwargs):
    net_code=kwargs.get('net_code',None)

    multiplier = kwargs.get('multiplier', 1.0)
    is_training=kwargs.get('is_training',True)
    num_class=kwargs.get('num_class',1000)
    label_smooth=kwargs.get('label_smooth', False)
    smooth_alpha=kwargs.get('smooth_alpha',0.1)
    mbv2base=kwargs.get('mbv2base',True)
    use_aux_head = kwargs.get('use_aux_head', False)
    aux_head_weight = kwargs.get('aux_head_weight', 0.4)
    has_dropout = kwargs.get('has_dropout', False)
    dropout_ratio = kwargs.get('dropout_ratio', 0.2)

    setting_params=get_setting_params(**kwargs)
    net_params=get_eatnet_param(net_code, _mbv2base=mbv2base)

    num_stage=net_params['num_stage']
    repeat_num=net_params['repeat_num']
    input_output_filter=net_params['input_output_filter']
    first_stride=net_params['first_stride']
    is_id_skip=net_params['is_id_skip']
    kernel_size=net_params['kernel_size']
    filter_depth=net_params['filter_depth']
    conv_ops = net_params['conv_ops']
    assert num_stage==7

    # for classification
    dilate_list=[1,1,1,1,1,1,1] # for classification
    with_dilate_list=[False, False, False, False,False,False,False]

    data = mx.sym.Variable(name='data')

    # head
    head_data=add_head_block(data=data,
                             num_filter=32,
                             setting_params=setting_params,
                             multiplier=multiplier,
                             kernel=(3, 3),
                             stride=(2, 2),
                             pad=(1, 1),
                             name="eatnet_head")

    stage0_data, _=add_stage_backbone_block(data=head_data,
                                            conv_ops = conv_ops,
                                            setting_params=setting_params,
                                            stage=0,
                                            repeat_num=repeat_num,
                                            input_output_filter=input_output_filter,
                                            first_stride=first_stride,
                                            is_id_skip=is_id_skip,
                                            kernel_size=kernel_size,
                                            filter_depth=filter_depth,
                                            multiplier=multiplier,
                                            dilate=dilate_list[0],
                                            with_dilate=with_dilate_list[0],
                                            name="eatnet")

    stage1_data, _=add_stage_backbone_block(data=stage0_data,
                                            conv_ops=conv_ops,
                                            setting_params=setting_params,
                                            stage=1,
                                            repeat_num=repeat_num,
                                            input_output_filter=input_output_filter,
                                            first_stride=first_stride,
                                            is_id_skip=is_id_skip,
                                            kernel_size=kernel_size,
                                            filter_depth=filter_depth,
                                            multiplier=multiplier,
                                            dilate=dilate_list[1],
                                            with_dilate=with_dilate_list[1],
                                            name="eatnet")

    stage2_data, _ = add_stage_backbone_block(data=stage1_data,
                                              conv_ops=conv_ops,
                                              setting_params=setting_params,
                                              stage=2,
                                              repeat_num=repeat_num,
                                              input_output_filter=input_output_filter,
                                              first_stride=first_stride,
                                              is_id_skip=is_id_skip,
                                              kernel_size=kernel_size,
                                              filter_depth=filter_depth,
                                              multiplier=multiplier,
                                              dilate=dilate_list[2],
                                              with_dilate=with_dilate_list[2],
                                              name="eatnet")
    stage3_data, _ = add_stage_backbone_block(data=stage2_data,
                                              conv_ops=conv_ops,
                                              setting_params=setting_params,
                                              stage=3,
                                              repeat_num=repeat_num,
                                              input_output_filter=input_output_filter,
                                              first_stride=first_stride,
                                              is_id_skip=is_id_skip,
                                              kernel_size=kernel_size,
                                              filter_depth=filter_depth,
                                              multiplier=multiplier,
                                              dilate=dilate_list[3],
                                              with_dilate=with_dilate_list[3],
                                              name="eatnet")
    stage4_data, _ = add_stage_backbone_block(data=stage3_data,
                                              conv_ops=conv_ops,
                                              setting_params=setting_params,
                                              stage=4,
                                              repeat_num=repeat_num,
                                              input_output_filter=input_output_filter,
                                              first_stride=first_stride,
                                              is_id_skip=is_id_skip,
                                              kernel_size=kernel_size,
                                              filter_depth=filter_depth,
                                              multiplier=multiplier,
                                              dilate=dilate_list[4],
                                              with_dilate=with_dilate_list[4],
                                              name="eatnet")
    if use_aux_head:
        softmax_aux = auxiliary_head(data=stage4_data,
                                    aux_head_weight=aux_head_weight,
                                    setting_params=setting_params,
                                    num_class=num_class)

    stage5_data, _ = add_stage_backbone_block(data=stage4_data,
                                              conv_ops=conv_ops,
                                              setting_params=setting_params,
                                              stage=5,
                                              repeat_num=repeat_num,
                                              input_output_filter=input_output_filter,
                                              first_stride=first_stride,
                                              is_id_skip=is_id_skip,
                                              kernel_size=kernel_size,
                                              filter_depth=filter_depth,
                                              multiplier=multiplier,
                                              dilate=dilate_list[5],
                                              with_dilate=with_dilate_list[5],
                                              name="eatnet")

    stage6_data, _ = add_stage_backbone_block(data=stage5_data,
                                              conv_ops=conv_ops,
                                              setting_params=setting_params,
                                              stage=6,
                                              repeat_num=repeat_num,
                                              input_output_filter=input_output_filter,
                                              first_stride=first_stride,
                                              is_id_skip=is_id_skip,
                                              kernel_size=kernel_size,
                                              filter_depth=filter_depth,
                                              multiplier=multiplier,
                                              dilate=dilate_list[6],
                                              with_dilate=with_dilate_list[6],
                                              name="eatnet")
    stage6_data_conv1x1 = add_conv_1x1(data=stage6_data,
                                       num_filter=int(1280 * multiplier) if multiplier > 1.0 else 1280,
                                       setting_params=setting_params,
                                       name='eatnet')

    if has_dropout:
        stage6_data_conv1x1 = mx.symbol.Dropout(stage6_data_conv1x1, p = dropout_ratio)

    softmax = add_fc_cls_block(data=stage6_data_conv1x1,
                               label_smooth=label_smooth,
                               smooth_alpha=smooth_alpha,
                               num_class=num_class,
                               is_training=is_training,
                               name='eatnet')
    
    if use_aux_head:
        output = mx.sym.Group([softmax, softmax_aux])
        return output
    else:
        return softmax

