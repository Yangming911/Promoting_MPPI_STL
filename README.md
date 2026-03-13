# STL PI Motion Planner

The STL PI Motion Planner is a framework designed to efficiently solve optimal control problems with Signal Temporal 
Logic (STL) cost functions. STL is commonly used to specify desired behaviors in dynamic systems because it allows for 
clear, modular, and flexible spatio-temporal specifications. Traditional optimization methods often struggle with 
STL-based problems due to scaling issues and nondifferentiable terms. This planner introduces a novel sampling-based 
method utilizing model predictive path integral control to overcome these challenges. This repository contains the 
implementation of the STL-PI planner as well as several state-of-the-art solvers.

---
## Installation

1. Create a new conda environment, e.g.,  `conda create -n stl_pi_planner python=3.8`. Activate this environment.
2. This project requires `libstdcxx-ng=13.2.0` from the `conda-forge` channel. Please ensure it is installed in your Conda environment: 
`conda install -c conda-forge libstdcxx-ng=13.2.0`
3. Install some C++ dependencies:
    * `sudo apt-get update`
    * `sudo apt-get install build-essential libomp-dev libeigen3-dev pybind11-dev ffmpeg`
4. Install the package: `pip install .` You can also run `pip install -v .` to print debug information.

---
## Run the motion planner
You can run the following scrips:
* `run_problem`: Runs a problem
* `compare_solvers`: Performs a numerical solver comparison
* `parameter_optimization`: Runs a parameter optimization for all solvers
