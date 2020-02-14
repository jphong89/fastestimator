# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Tuple, Iterable

import tensorflow as tf
from tensorflow.python.keras import layers, Model

def UNet(input_shape: Tuple[int, int, int] = (128, 128, 3),
         dropout: float = 0.5,
         nchannels: Iterable[int] = (64, 128, 256, 512, 1024),
         nclasses: int = 1,
         bn: str = None,
         activation: str = 'relu',
         upsampling: str = 'bilinear',
         dilation_rates: Iterable[int] = (1, 1, 1, 1, 1),
         residual: str = None):

    assert dropout is None or 0 <= dropout <= 1, "Invalid value for dropout parameter (None or 0 to 1 only)"

    assert bn in [None, "before", "after"], "Invalid bn parameter value"

    assert len(nchannels) >= 2, "At least 2 channels necessary for UNet"

    assert len(nchannels) == len(dilation_rates), "len(nchannels) should be the same as len(dilation_rates)"

    assert residual in [None, 'enc', 'dec', 'both'], 'Wrong argument specified for residual'

    # Handle callable activations as well
    if isinstance(activation, str):
        act = activation
    else:
        act = None

    conv_config = {'activation': act, 'padding': 'same', 'kernel_initializer': 'he_normal'}

    inputs = layers.Input(input_shape)
    inp = inputs

    levels = []

    # Contracting blocks
    residual_enc_flag = residual in ['enc', 'both']
    for idx, nc in enumerate(nchannels[:-1]):

        if idx == len(nchannels) - 2:
            d = dropout
        else:
            d = None

        C, C_pooled = _conv_block(inputs,
                                  nc,
                                  3,
                                  conv_config,
                                  pooling=2,
                                  dropout=d,
                                  bn=bn,
                                  activation=activation,
                                  dilation_rate=dilation_rates[idx],
                                  residual=residual_enc_flag)
        levels.append((C, C_pooled))
        inputs = C_pooled

    residual_dec_flag = residual in ['dec', 'both']
    # Expanding blocks
    inp1, inp2 = levels[-1][1], levels[-1][0]
    for idx, nc in enumerate(reversed(nchannels[1:])):
        if idx == 0:
            d = dropout
            dilation = dilation_rates[-1]
        else:
            d = None
            dilation = None

        D = _up_concat(inp1,
                       inp2,
                       2,
                       (nchannels[-1 - idx], nchannels[-2 - idx]),
                       3,
                       conv_config,
                       d,
                       bn=bn,
                       activation=activation,
                       upsampling=upsampling,
                       dilation=dilation,
                       residual=residual_dec_flag)
        if idx != len(nchannels) - 2:
            inp1, inp2 = D, levels[-2 - idx][0]

    C_end1, _ = _conv_block(D, 64, 3, conv_config, bn=bn, activation=activation)

    if bn:
        if bn == 'before':
            act = conv_config['activation']
            conv_config['activation'] = None

    C_end2 = layers.Conv2D(2, 3, **conv_config)(C_end1)

    if bn:
        C_end2 = layers.BatchNormalization()(C_end2)

        if bn == 'before':
            if act:
                C_end2 = layers.Activation(act)(C_end2)
            else:
                C_end2 = activation(C_end2)

    y_dist = layers.Conv2D(nclasses, 1, activation='sigmoid')(C_end2)

    model = Model(inputs=inp, outputs=y_dist)

    return model


def _conv_block(inp,
                nchannels,
                window,
                config,
                pooling=None,
                dropout=None,
                bn=False,
                activation=None,
                dilation_rate=1,
                residual=False):

    if bn and bn == 'before':
        act = config['activation']
        config['activation'] = None

    if residual:
        inp = layers.Conv2D(nchannels, 1, dilation_rate=dilation_rate, **config)(inp)

    conv1 = layers.Conv2D(nchannels, window, dilation_rate=dilation_rate, **config)(inp)

    if bn:
        conv1 = layers.BatchNormalization()(conv1)

        if bn == 'before':
            if act:
                conv1 = layers.Activation(act)(conv1)
            else:
                conv1 = activation(conv1)

    if residual:
        conv1 = inp + conv1

        conv1 = layers.Conv2D(nchannels, 1, dilation_rate=dilation_rate, **config)(conv1)

    conv2 = layers.Conv2D(nchannels, window, dilation_rate=dilation_rate, **config)(conv1)

    if bn:
        conv2 = layers.BatchNormalization()(conv2)

        if bn == 'before':
            if act:
                conv2 = layers.Activation(act)(conv2)
            else:
                conv2 = activation(conv2)

            config['activation'] = act  # python dicts are reference based

    if dropout:
        conv2 = layers.Dropout(dropout)(conv2)

    if residual:
        conv2 = layers.Conv2D(nchannels, 1, dilation_rate=dilation_rate, **config)(conv2)
        conv2 = conv1 + conv2

    if pooling:
        pooled = layers.MaxPooling2D(pool_size=(pooling, pooling))(conv2)
    else:
        pooled = None

    return conv2, pooled


def _upsample(inp,
              factor,
              nchannels,
              config,
              bn=None,
              activation=None,
              upsampling='bilinear',
              residual=False):

    if residual:
        r1 = layers.UpSampling2D(size=(factor, factor), interpolation=upsampling)(inp)
        up = layers.Conv2D(nchannels, factor, **config)(r1)
        r2 = layers.Conv2DTranspose(nchannels, factor, strides=(factor, factor), padding='same')(inp)

    else:
        if upsampling in ['bilinear', 'nearest']:
            up = layers.UpSampling2D(size=(factor, factor), interpolation=upsampling)(inp)
            up = layers.Conv2D(nchannels, factor, **config)(up)
        elif upsampling == 'conv':
            up = layers.Conv2DTranspose(nchannels, factor, strides=(factor, factor), padding='same')(inp)
        else:
            raise NotImplementedError

    if bn and bn == 'before':
        act = config['activation']
        config['activation'] = None

    if bn:
        up = layers.BatchNormalization()(up)

        if bn == 'before':
            if act:
                up = layers.Activation(act)(up)
            else:
                up = activation(up)

            config['activation'] = act

    if residual:
        up = up + r2

    return up


def _up_concat(conv_pooled,
               conv,
               factor,
               nchannels,
               window,
               config,
               dropout=None,
               bn=None,
               activation=None,
               upsampling='bilinear',
               dilation=1,
               residual=False):

    assert len(nchannels) == 2

    F, _ = _conv_block(conv_pooled, nchannels[0], window, config, bn=bn, dropout=dropout, activation=activation, dilation_rate=dilation if dilation else 1, residual=residual)

    upsampled = _upsample(F,
                          factor,
                          nchannels[1],
                          config,
                          bn=bn,
                          activation=activation,
                          upsampling=upsampling,
                          residual=residual)

    feat = layers.concatenate([conv, upsampled], axis=3)

    return feat
