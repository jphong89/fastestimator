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
"""WGANGP example using MNIST data set."""
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

import fastestimator as fe
from fastestimator.estimator.trace import ModelSaver
from fastestimator.network.loss import Loss
from fastestimator.network.model import FEModel, ModelOp
from fastestimator.util.op import TensorOp

DIM = 64

class GLoss(Loss):
    """Compute generator loss."""
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        return -tf.reduce_mean(data, axis=[1,2,3])


class DLoss(Loss):
    """Compute discrimator loss."""
    def __init__(self, inputs, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state):
        true, fake = data
        total_loss = tf.reduce_mean(true, axis=[1,2,3]) - tf.reduce_mean(fake, axis=[1,2,3])
        return total_loss

class RandomInterpolate(TensorOp):
    def forward(self, data, state):
        pass

class ComputeGradientPenalty(TensorOp):
    pass


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(DIM, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

class Myrescale(TensorOp):
    """Scale image values from uint8 to float32 between -1 and 1."""
    def forward(self, data, state):
        data = tf.cast(data, tf.float32)
        data = (data - 127.5) / 127.5
        return data


def get_estimator(batch_size=256, epochs=50, model_dir=tempfile.mkdtemp()):
    # prepare data
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    data = {"train": {"x": np.expand_dims(x_train, -1)}}
    pipeline = fe.Pipeline(batch_size=batch_size, data=data, ops=Myrescale(inputs="x", outputs="x"))
    # prepare model
    g_femodel = FEModel(model_def=make_generator_model,
                        model_name="gen",
                        loss_name="gloss",
                        optimizer=tf.optimizers.Adam(1e-4))
    d_femodel = FEModel(model_def=make_discriminator_model,
                        model_name="disc",
                        loss_name="dloss",
                        optimizer=tf.optimizers.Adam(1e-4))
    network = fe.Network(ops=[
        ModelOp(inputs=lambda: tf.random.normal([batch_size, DIM]), model=g_femodel),
        ModelOp(model=d_femodel, outputs="pred_fake"),
        ModelOp(inputs="x", model=d_femodel, outputs="pred_true"),
        GLoss(inputs=("pred_fake"), outputs="gloss"),
        DLoss(inputs=("pred_true", "pred_fake"), outputs="dloss")
    ])
    # prepare estimator
    traces = [ModelSaver(model_name='gen', save_dir=model_dir, save_freq=5)]
    estimator = fe.Estimator(network=network, pipeline=pipeline, epochs=epochs, traces=traces)
    return estimator


if __name__ == "__main__":
    est = get_estimator()
    est.fit()
