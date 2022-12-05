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


#include <benchmark/benchmark.h>
#include <unistd.h>

#include <random>

#include "lite/kernels/arm/conv_compute.h"
#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif
#include "lite/tests/benchmark/src/convolution_fp16_configs.h"

template <class Tin,
          class Tout,
          paddle::lite::PrecisionType Ptype,
          paddle::lite::PrecisionType OutType>
void bench_conv(const benchmark::State& state_in, const char* net) {
  // const in parameter is used to pass CI system
  // because google bench mark must work with a `benchmark::State &`
  // we do a const cast here
  benchmark::State& state = const_cast<benchmark::State&>(state_in);

  const int64_t batch_size = state.range(0);
  const int64_t input_height = state.range(1);
  const int64_t input_width = state.range(2);
  const int64_t groups = state.range(3);
  const int64_t groups_output_channels = state.range(4);
  const int64_t groups_input_channels = state.range(5);
  const int64_t kernel_height = state.range(6);
  const int64_t kernel_width = state.range(7);
  const int64_t padding_top = state.range(8);
  const int64_t padding_down = state.range(9);
  const int64_t padding_left = state.range(10);
  const int64_t padding_right = state.range(11);
  const int64_t s_h = state.range(12);
  const int64_t s_w = state.range(13);
  const int stride_h = s_h;
  const int stride_w = s_w;
  const int64_t dilation_h = state.range(14);
  const int64_t dilation_w = state.range(15);
  
  //compute dim info of output tensor 
  const int64_t effective_kernel_height = (kernel_height - 1) * dilation_h + 1;
  const int64_t effective_kernel_width = (kernel_width - 1) * dilation_w + 1;
  const int64_t output_height =
      (input_height + padding_top + padding_down - effective_kernel_height) / stride_h + 1;
  const int64_t output_width =
      (input_width + padding_left + padding_right - effective_kernel_width) / stride_w + 1;
  //generate random number to fill tensor
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(-10, 10), std::ref(rng));
  using paddle::lite::DDim;
  using paddle::lite::Tensor;

  //x : [batch_size, groups * groups_input_channels, input_height, input_width]
  //filter : [groups * groups_output_channels, group_input_channels, kernel_height, kernel_width]
  //output : [batch_size, groups * groups_output_channels, output_height, output_width]
  Tensor x, filter, bias, output;
  x.Resize(DDim(
      {batch_size, groups * groups_input_channels, input_height, input_width}));
  std::generate(x.mutable_data<Tin>(),
                x.mutable_data<Tin>() + x.numel(),
                std::ref(input_rng));
  filter.Resize(DDim({groups * groups_output_channels,
                      groups_input_channels,
                      kernel_height,
                      kernel_width}));
  std::generate(filter.mutable_data<Tin>(),
                filter.mutable_data<Tin>() + filter.numel(),
                std::ref(input_rng));
  bias.Resize(DDim({groups * groups_output_channels}));
  std::generate(bias.mutable_data<float>(),
                bias.mutable_data<float>() + bias.numel(),
                std::ref(input_rng));
  output.Resize(DDim({batch_size,
                      groups * groups_output_channels,
                      output_height,
                      output_width}));
  output.mutable_data<Tout>();

  // initial param of conv
  paddle::lite::kernels::arm::ConvCompute<Ptype, OutType> conv_compute;
  paddle::lite::operators::ConvParam param;
  param.x = &x;
  param.bias = &bias;
  param.filter = &filter;
  param.output = &output;

  const size_t pad_top = padding_top;
  const size_t pad_bottom = padding_down;
  const size_t pad_left = padding_left;
  const size_t pad_right = padding_right;
  auto pd = std::make_shared<std::vector<int>>();
  pd->push_back(pad_top);
  pd->push_back(pad_bottom);
  pd->push_back(pad_left);
  pd->push_back(pad_right);
  param.paddings = pd;

  param.strides = std::vector<int>{stride_h, stride_w};
  param.groups = groups;
  auto dl = std::make_shared<std::vector<int>>();
  dl->push_back(dilation_h);
  dl->push_back(dilation_w);
  param.dilations = dl;

  conv_compute.SetParam(param);

  // set context of conv kernel
  auto ctx1 = paddle::lite::ContextScheduler::Global().NewContext(
      paddle::lite_api::TargetType::kARM);
  auto& ctx = ctx1->As<paddle::lite::ARMContext>();
  ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(
                     paddle::lite_api::PowerMode::LITE_POWER_HIGH),
                 1);

  conv_compute.SetContext(std::move(ctx1));
  conv_compute.PrepareForRun();
  
  // start benchmark test
  for (auto _ : state) {
    conv_compute.Launch();
  }

  // get gop of different fp16 conv kernel
  state.counters["OPS"] = benchmark::Counter(
      uint64_t(state.iterations()) * 
          batch_size * groups * groups_output_channels * output_height * output_width *
          2 * groups_input_channels * kernel_height * kernel_width,
      benchmark::Counter::kIsRate);
}

constexpr static auto f16_conv =
    bench_conv<float16_t, float16_t, PRECISION(kFP16), PRECISION(kFP16)>;
BENCHMARK_CONVOLUTION(f16_conv)

BENCHMARK_MAIN();
