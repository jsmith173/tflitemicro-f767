/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#define USE_FLOAT

#include "main_functions.h"
#include "main.h"
#include "img_array.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_quant_data.h"

#define ARENA_SIZE (440*1024)

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = ARENA_SIZE;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
int ai_setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model_quant_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 0;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return 0;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
  
  // ai_setup end: Blue LED
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_SET);   
  
  return 1;
}

template <typename T>
int argmax() 
{
  int size = output->bytes; 
  int div_size = 1;
  if (output->type == kTfLiteFloat32)
    div_size = sizeof(float);
  int num_classes = size/div_size;
  
  int idx = 0;
  T vi, v = 0;
  for (uint32_t i = 0; i < num_classes; i++) {
    T* output_data = tflite::GetTensorData<T>(output);
    for (int i = 0; i < num_classes; i++) {
      vi = output_data[i];
      if (vi > v){
        idx = i;
        v = vi;
      }
    }
  }
  return idx;
}

int ai_result[32];

// The name of this function is important for Arduino compatibility.
int ai_loop() {
  int tensor_size = input->bytes;
  int dim_size = input->dims->size;
  int dim_1 = input->dims->data[0];
  int dim_h = input->dims->data[1];
  int dim_w = input->dims->data[2];
  int dim_pic = 1;
  if (dim_size == 4)
   dim_pic = input->dims->data[3];
  int N = dim_w*dim_h;
  float *input_data_float=NULL;
  uint8_t* input_data_uint8=NULL;
  int passed, data, k;
  uint8_t ui8Data;
  
  input_data_uint8 = tflite::GetTensorData<uint8_t>(input);

  // Copy the buffer to input tensor
  k = 0;
  for (int i = 0; i < N; i++) {
   data = img_array[i];
   for (int j = 0; j < 3; j++) {
	ui8Data = data & 0xFF;
    float x = ui8Data;
    uint8_t x_quantized = x / input->params.scale + input->params.zero_point;
    input_data_uint8[k++] = x_quantized;
	data >>= 8;
   }
  }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n");
    return -1;
  }

  // Get output and ArgMax
  int idx;
  if (output->type == kTfLiteFloat32)
   idx = argmax<float>();
  else
   idx = argmax<uint8_t>();
  
  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
  
  return idx;
}
