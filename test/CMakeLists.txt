include(../cmake/cuda.cmake)

# find lib GoogleTest, glog
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(third_party_lib glog::glog GTest::gtest)

# test root dir
aux_source_directory(../test TEST_ROOT_DIR)

# test sub dir
aux_source_directory(../test/test_base TEST_BASE_DIR)
aux_source_directory(../test/test_model TEST_MODEL_DIR)
aux_source_directory(../test/test_tensor TEST_TENSOR_DIR)
aux_source_directory(../test/test_op TEST_OP_DIR)

# exe 
add_executable(test_llm 
  ${TEST_ROOT_DIR}
  ${TEST_BASE_DIR}
  ${TEST_MODEL_DIR}
  ${TEST_TENSOR_DIR}
  ${TEST_OP_DIR}
)

# lib link glog,gtest
target_link_libraries(test_llm ${third_party_lib})

# lib inc glog,gtest
target_include_directories(test_llm PUBLIC 
  ${glog_INCLUDE_DIR}
  ${GTest_INCLUDE_DIR}
)
# cachehitML inc
target_include_directories(test_llm PUBLIC
  ../cachehitML/include
)
# link dir
target_link_directories(test_llm PUBLIC
  ${PROJECT_SOURCE_DIR}/lib
)

# if llama3 or qwen2 support OPEN
if(LLAMA3_SUPPORT OR QWEN2_SUPPORT)
  message(STATUS "LINK LLAMA3 AND QWEN2 SUPPORT")
  find_package(absl REQUIRED)
  find_package(re2 REQUIRED)
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif()

target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




