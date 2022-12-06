// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

//x : [batch_size, groups * groups_input_channels, input_height, input_width]
//filter : [groups * group_output_channels, group_input_channels, kernel_height, kernel_width]
//output : [batch_size, groups * group_output_channels, output_height, output_width]
// Depthwise GCout = GCin = 1
// TEST
static void DepthTest(benchmark::internal::Benchmark* b) {
  b->ArgNames(
      {"N", "H", "W", "G", "GCout", "GCin", "KH", "KW", "PU", "PD", "PL", "PR", "SH", "SW", "DH", "DW"});

  /*       N   H    W   G  GCout  GCin  KH  KW  PU  PD  PL  PR  SH  SW  DH  DW*/
  //depthwise 
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 2, 1, 1});//gemm
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //direct sw=2
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({1, 224, 224, 1, 166, 512, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});//gemm
  //direct sw=1 && winograd
  b->Args({1, 224, 224, 1, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 2, 16, 32, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //gemm 
}

static void DirectS2Test(benchmark::internal::Benchmark* b) {
  b->ArgNames(
      {"N", "H", "W", "G", "GCout", "GCin", "KH", "KW", "PU", "PD", "PL", "PR", "SH", "SW", "DH", "DW"});

  /*       N   H    W   G  GCout  GCin  KH  KW  PU  PD  PL  PR  SH  SW  DH  DW*/
  //depthwise 
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 2, 1, 1});//gemm
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //direct sw=2
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({1, 224, 224, 1, 166, 512, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});//gemm
  //direct sw=1 && winograd
  b->Args({1, 224, 224, 1, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 2, 16, 32, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //gemm 
}

static void DirectS1Test(benchmark::internal::Benchmark* b) {
  b->ArgNames(
      {"N", "H", "W", "G", "GCout", "GCin", "KH", "KW", "PU", "PD", "PL", "PR", "SH", "SW", "DH", "DW"});

  /*       N   H    W   G  GCout  GCin  KH  KW  PU  PD  PL  PR  SH  SW  DH  DW*/
  //depthwise 
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 5, 5, 1, 1, 1, 1, 1, 2, 1, 1});//gemm
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 16, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //direct sw=2
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});
  b->Args({1, 224, 224, 1, 166, 512, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1});//gemm
  //direct sw=1 && winograd
  b->Args({1, 224, 224, 1, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 1, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  b->Args({1, 224, 224, 2, 16, 32, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1});//gemm
  //gemm 
}

#define BENCHMARK_CONVOLUTION(conv_fn)                                     \
  BENCHMARK_CAPTURE(conv_fn, test, "Test")                                 \
      ->Apply(Test)                                                        \
      ->UseRealTime();                                                     
