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
#include "gsl/gsl"
#include "core/providers/pim/helper/pim_interface.h"
#include "core/providers/pim/pim_execution_provider.h"

#define LUT_SIZE 65536

namespace lut_env_vars {
static const std::string kLutPath = "ORT_PIM_LUT_PATH";
}  // namespace tensorrt_env_vars

namespace onnxruntime {

class LutHelper {
 public:
  LutHelper(
    // Bfloat16* lut_data,
    // const std::string& function_name
    // bool is_table_exist, 
    size_t lut_ops_num,
    Bfloat16** lut_ptr_arr_ptr
    ) 
{

//Pre-existed file list
/*Abs(), Div(), Erf(), Log(), Neg(), Pow2(), Relu(), Sigmoid(), Sqrt(), Tanh() */
    std::vector<std::string> lut_tables = {"Abs","Erf","Log","Neg","Relu","Sigmoid","Sqrt","Tanh"};
    //                                        0,     1,     2,    3,    4,       5,      6,      7

    Path lut_base_dir;
    Path lut_func_file;
    std::string function_name;
    lut_base_dir = Path::Parse(ToPathString(Env::Default().GetEnvironmentVar(lut_env_vars::kLutPath)));

  for(size_t i=0; i<=LUT_OPS_NUM; i++){
    function_name = lut_tables[i];
    lut_func_file = lut_base_dir / Path::Parse(MakeLutFileName(function_name));

    if (lut_base_dir.IsEmpty()) {
      ORT_NOT_IMPLEMENTED("[PIM Execution Provider] ORT_PIM_LUT_PATH isn't configured! Please use ORT_PIM_LUT_PATH to specify pre-existed lut data path");
    }
    //For future uses
    if (!FileExistanceCheck(function_name, lut_tables)){
      status_ = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Target function's LUT is not generated");
    }
    else{
      const PathString file_path_str = lut_func_file.ToPathString();
      // Bfloat16* lut_ptr = (Bfloat16*)(pim_malloc(LUT_SIZE*sizeof(Bfloat16)));
      Bfloat16  x =0x3000;
      Bfloat16* lut_ptr = &x;

        if (lut_ptr == NULL) {
          perror("PIM Memory Allocation Failed");
          exit(1);
        } 
      readFile(file_path_str.c_str(), lut_ptr, LUT_SIZE);
      // for(size_t i=0; i<LUT_SIZE; i++){
      //   std::cout<<"16'h"<<std::hex<<lut_ptr[i]<<std::endl;
      // }
      LOGS_DEFAULT(WARNING) << "PIM LUT tensor is allocated and initialized! Op type:" <<function_name;
      lut_ptr_arr_ptr[i] = lut_ptr;
      // lut_ptr_ = lut_ptr;

    }
  }
}

  Status State() const { return status_; }
  // Bfloat16** lut_ptrs() { return result_ptr_lists_; }

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

void readFile(const char * fname, Bfloat16* array, unsigned length)
  {
      std::ifstream fin(fname, std::ios::binary);
      fin.seekg(0, std::ios_base::end);
      unsigned file_len = fin.tellg();
      if (file_len != length * sizeof(short))
      {
          std::cout << "Error: file length: " << file_len 
                    << "  expected: " << length * sizeof(short) << std::endl;
          return; 
      }
      
      fin.seekg(0, std::ios_base::beg);
      fin.read( (char *) array, file_len);
      fin.close();
  }

  Status status_;
  // Bfloat16* lut_ptr_;
  // Bfloat16* result_ptr_lists_[LUT_OPS_NUM];
};

}  // namespace onnxruntime