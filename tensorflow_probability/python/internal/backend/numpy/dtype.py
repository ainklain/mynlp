# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Numpy implementations of TensorFlow dtype related."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils


__all__ = [
    'as_dtype',
    'bool',
    'complex',
    'complex128',
    'complex64',
    'double',
    'float16',
    'float32',
    'float64',
    'half',
    'int16',
    'int32',
    'int64',
    'int8',
    'string',
    'uint16',
    'uint32',
    'uint64',
    'uint8',
    # 'as_string',
    # 'bfloat16',
    # 'dtypes',
    # 'qint16',
    # 'qint32',
    # 'qint8',
    # 'quint16',
    # 'quint8',
]


# --- Begin Public Functions --------------------------------------------------

as_dtype = utils.copy_docstring(
    tf.as_dtype,
    lambda type_value: np.dtype(type_value).type)

bool = np.bool  # pylint: disable=redefined-builtin

complex = np.complex  # pylint: disable=redefined-builtin

complex128 = np.complex128

complex64 = np.complex64

double = np.double

float16 = np.float16

float32 = np.float32

float64 = np.float64

half = np.half

int16 = np.int16

int32 = np.int32

int64 = np.int64

int8 = np.int8

string = np.str

uint16 = np.uint16

uint32 = np.uint32

uint64 = np.uint64

uint8 = np.uint8
