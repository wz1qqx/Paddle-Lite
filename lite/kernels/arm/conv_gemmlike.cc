// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/conv_gemmlike.h"
#include <vector>
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->Param<param_t>();
  /// update scale
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  for (auto& ws : w_scale_) {
    ws *= input_scale;
  }
}

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->Param<param_t>();
  /// update scale
  w_scale_ = param.weight_scale;
  if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
    LOG(FATAL) << "weights scale size must equal to filter size";
    return;
  }
  if (w_scale_.size() == 1) {
    for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
      w_scale_.push_back(w_scale_[0]);
    }
  }
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  for (auto& ws : w_scale_) {
    ws = ws * input_scale / output_scale;
  }
  //!  update bias
  if (param.bias) {
    bias_.Resize(param.bias->dims());
    auto ptr = bias_.mutable_data<float>();
    auto ptr_in = param.bias->data<float>();
    for (int i = 0; i < bias_.numel(); ++i) {
      ptr[i] = ptr_in[i] / param.output_scale;
    }
    flag_trans_bias_ = true;
  }
  //! update relu6 parameter
  if (param.activation_param.active_type == lite_api::ActivationType::kRelu6) {
    param.activation_param.Relu_clipped_coef =
        param.activation_param.Relu_clipped_coef / param.output_scale;
  }
  //! update hardswish parameter
  if (param.activation_param.active_type ==
      lite_api::ActivationType::kHardSwish) {
    param.activation_param.hard_swish_scale =
        param.activation_param.hard_swish_scale / param.output_scale;
    param.activation_param.hard_swish_offset =
        param.activation_param.hard_swish_offset / param.output_scale;
    param.activation_param.hard_swish_threshold =
        param.activation_param.hard_swish_threshold / param.output_scale;
  }
}

PROFILE_INFO(kFloat, kFloat);

template <>
void GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<float>();
  if (flag_trans_weights_) {
    weights = weights_.data<float>();
  }
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  if (flag_1x1gemm_) {
    lite::arm::math::conv1x1s1_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv1x1s1_gemm_fp32")
  } else {
    lite::arm::math::conv_im2col_gemm(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv_im2col_gemm_fp32")
  }
}

PROFILE_INFO(kInt8, kFloat);

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<int8_t>();
  if (flag_trans_weights_) {
    weights = weights_.data<int8_t>();
  }
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  if (flag_1x1gemm_) {
    lite::arm::math::conv1x1s1_gemm_int8(din,
                                         dout,
                                         bs,
                                         oc,
                                         oh,
                                         ow,
                                         ic,
                                         ih,
                                         iw,
                                         weights,
                                         bias,
                                         param,
                                         &ctx,
                                         w_scale_.data());
    KERNEL_FUNC_NAME("conv1x1s1_gemm_int8")
  } else {
    lite::arm::math::conv_im2col_gemm_int8(din,
                                           dout,
                                           bs,
                                           oc,
                                           oh,
                                           ow,
                                           ic,
                                           ih,
                                           iw,
                                           weights,
                                           bias,
                                           param,
                                           &ctx,
                                           w_scale_.data());
    KERNEL_FUNC_NAME("conv_im2col_gemm_int8")
  }
}

PROFILE_INFO(kInt8, kInt8)

template <>
void GemmLikeConv<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<int8_t>();
  if (flag_trans_weights_) {
    weights = weights_.data<int8_t>();
  }
  auto bias = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float>();
  }
  auto din = param.x->data<int8_t>();
  auto dout = param.output->mutable_data<int8_t>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  if (flag_1x1gemm_) {
    lite::arm::math::conv1x1s1_gemm_int8(din,
                                         dout,
                                         bs,
                                         oc,
                                         oh,
                                         ow,
                                         ic,
                                         ih,
                                         iw,
                                         weights,
                                         bias,
                                         param,
                                         &ctx,
                                         w_scale_.data());
    KERNEL_FUNC_NAME("conv1x1s1_gemm_int8")
  } else {
    lite::arm::math::conv_im2col_gemm_int8(din,
                                           dout,
                                           bs,
                                           oc,
                                           oh,
                                           ow,
                                           ic,
                                           ih,
                                           iw,
                                           weights,
                                           bias,
                                           param,
                                           &ctx,
                                           w_scale_.data());
    KERNEL_FUNC_NAME("conv_im2col_gemm_int8")
  }
}

#ifdef ENABLE_ARM_FP16
template <>
void GemmLikeConv<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  //ReInitWhenNeeded();
    auto& param = this->template Param<param_t>();
    CHECK(this->ctx_);
    auto& ctx = this->ctx_->template As<ARMContext>();
    auto x_dims = param.x->dims();
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();
    if (last_shape_ == x_dims) {
      return;
    }

    int ic = x_dims[1];
    int ih = x_dims[2];
    int iw = x_dims[3];  // nchw
    int oc = o_dims[1];
    int oh = o_dims[2];
    int ow = o_dims[3];
    int kh = w_dims[2];
    int kw = w_dims[3];

    auto paddings = *param.paddings;
    auto dilations = *param.dilations;

    int sw = param.strides[1];
    int sh = param.strides[0];
    int pw = paddings[2];
    int ph = paddings[0];
    int dw = dilations[1];
    int dh = dilations[0];

    bool pads_equal =
        ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

    int m = oc / param.groups;
    int k = ic * kh * kw / param.groups;
    int n = oh * ow;

    bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
    bool ks_equal = (sw == sh) && (kw == kh);
    //! select conv gemmlike kernel
    if (kw == 1 && sw == 1 && pw == 0 && kps_equal && pads_equal) {
      //! 1x1s1p0 gemmlike conv
      flag_1x1gemm_ = true;
    } else {
      //! im2col gemmlike conv
      flag_1x1gemm_ = false;
      workspace_size_ = k * n * sizeof(float);
    }
    if (!flag_trans_weights_ && n > 1 && m > 1) {
      if (param.filter->precision() == PrecisionType::kFP16) {
#ifdef ENABLE_ARM_FP16
        flag_trans_weights_ = false;
        // lite::arm::math::fp16::trans_gemm_weights_fp16(
        //     *(param.filter), weights_, param.groups, &ctx);
#else
        LOG(FATAL) << "FP16 conv must open ENABLE_ARM_FP16";
#endif
      } else {
        lite::arm::math::trans_gemm_weights<Ptype>(
            *(param.filter), weights_, param.groups, &ctx);
        flag_trans_weights_ = true;
      }
    } else if (n == 1 || m == 1) {
      flag_trans_weights_ = false;
    }
    last_shape_ = x_dims;
}

PROFILE_INFO(kFP16, kFP16)

template <>
void GemmLikeConv<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  auto weights = param.filter->data<float16_t>();
  if (flag_trans_weights_) {
    weights = weights_.data<float16_t>();
  }
  const float16_t* bias = param.bias ? param.bias->data<float16_t>() : nullptr;
  if (flag_trans_bias_) {
    bias = bias_.data<float16_t>();
  }
  auto din = param.x->data<float16_t>();
  auto dout = param.output->mutable_data<float16_t>();

  auto x_dims = param.x->dims();
  // auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int bs = x_dims[0];
  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];  // nchw
  int oc = o_dims[1];  // oc == kn
  int oh = o_dims[2];
  int ow = o_dims[3];
  if (flag_1x1gemm_) {
    lite::arm::math::fp16::conv1x1s1_gemm_fp16(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv1x1s1_gemm_fp16")
  } else {
    lite::arm::math::fp16::conv_im2col_gemm_fp16(
        din, dout, bs, oc, oh, ow, ic, ih, iw, weights, bias, param, &ctx);
    KERNEL_FUNC_NAME("conv_im2col_gemm_fp16")
  }
}

#endif
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
