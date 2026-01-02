// Copyright (c) Facebook, Inc. and affiliates.
// Minimal vision.cpp for deformable convolution only

#include <torch/extension.h>
#include "deformable/deform_conv.h"

#include <torch/extension.h>
#include "deformable/deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &tcd::deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &tcd::deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &tcd::deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &tcd::modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &tcd::modulated_deform_conv_backward,
      "modulated_deform_conv_backward");
}

