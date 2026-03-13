cmake_minimum_required(VERSION 3.4...3.18)
project(stl_pi_planner_c)

# Find Python
find_package(pybind11 REQUIRED)

# Ensure Python3 is available with the development components needed for pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find required libraries
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

# Create the bindings library
pybind11_add_module(stl_pi_planner_c MODULE python_binding/pybind_stl_pi_planner_c.cpp)

# Link with the required libraries
target_link_libraries(stl_pi_planner_c PUBLIC Eigen3::Eigen)
target_link_libraries(stl_pi_planner_c PRIVATE OpenMP::OpenMP_CXX)

# Ensure C++17 standard is used
set_target_properties(stl_pi_planner_c PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# Installation path for the Python module (ensuring it goes to the Python site-packages)
if(NOT DEFINED Python3_SITELIB)
    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
            OUTPUT_VARIABLE Python3_SITELIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
message(STATUS "Python site-packages directory: ${Python3_SITELIB}")

# Install into site-packages
install(TARGETS stl_pi_planner_c LIBRARY DESTINATION ${Python3_SITELIB})
