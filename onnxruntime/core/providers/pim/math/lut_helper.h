// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/common/path_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"

#include <fstream>
#include <string>
#include <iostream>
#include <experimental/filesystem>

namespace lut_env_vars {
static const std::string kLutPath = "ORT_PIM_LUT_PATH";
}  // namespace tensorrt_env_vars

namespace onnxruntime {

class LutHelper {
 public:
  LutHelper(
    const TensorShape& A,
    const TensorShape& B,
    const std::string& function_name
    // bool is_table_exist, 
    ) 
{
//Dimension check
//Can be differed by operation
    // ORT_ENFORCE(A.NumDimensions() == 2 || B.NumDimensions() == 1);
    // ORT_ENFORCE(B.NumDimensions() == 2);
    
//Check if shape is aligned at PIM's input size
    // bool is_aligned;
    // is_aligned = A->Shape()[];
    //  M_ = left.NumDimensions() == 2 ? left[1] : left[0];

//Pre-existed file list
/*Abs(), Div(), Erf(), Log(), Neg(), Pow2(), Relu(), Sigmoid(), Sqrt(), Tanh() */
    std::vector<std::string> lut_tables = {"Abs","Div","Erf","Log","Neg","Pow","Relu","Sigmoid","Sqrt","Tanh"};
    Path lut_base_dir;
    Path lut_func_file;

    lut_base_dir = Path::Parse(ToPathString(Env::Default().GetEnvironmentVar(lut_env_vars::kLutPath)));
    lut_func_file = lut_base_dir / Path::Parse(MakeLutFileName(function_name));

    if (lut_base_dir.IsEmpty()) {
      ORT_NOT_IMPLEMENTED("[PIM Execution Provider] ORT_PIM_LUT_PATH isn't configured! Please use ORT_PIM_LUT_PATH to specify pre-existed lut data path");
    }

    if (!FileExistanceCheck(function_name, lut_tables)){
      status_ = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Target function's LUT is not generated");
    }
    else{
      const PathString file_path_str = lut_func_file.ToPathString();
      int output_fd;
      ORT_THROW_IF_ERROR(Env::Default().FileOpenWr(file_path_str, output_fd));
      filedesc_ = output_fd;  
      //ORT_THROW_IF_ERROR(Env::Default().FileClose(output_fd));
    }
  }

  int64_t M() const { return M_; }
  int64_t N() const { return N_; }
  Status State() const { return status_; }
  int64_t FileDesc() const { return filedesc_; }

 private:
  bool FileExistanceCheck(const std::string& funct, std::vector<std::string>& check_tables) {
// Furture works
    auto check_func_it = find(check_tables.begin(), check_tables.end(), funct);
    if(check_func_it == check_tables.end()){
        return false;
    }
    else {
        return true;
    }
  }
  PathString MakeLutFileName(const std::string& funct) {
  auto make_valid_name = [](std::string name) {
    std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    return name;
  };

  return path_utils::MakePathString(make_valid_name(funct),".dat");
}


 private:
  int64_t M_;
  int64_t N_;
  Status status_;
//   std::unique_ptr<std::fstream> fileptr_;
  int64_t filedesc_;
};

}  // namespace onnxruntime
