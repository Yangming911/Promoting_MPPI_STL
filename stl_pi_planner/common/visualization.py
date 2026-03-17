import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from stl_pi_planner.systems import DoubleIntegrator, SingleIntegrator
from stl_pi_planner.systems.linear import PointMass
from stl_pi_planner.systems.nonlinear import Unicycle, Bicycle


def visualize_solution(x, u, x0, K, spec_name, plot_scenario_fct, system, solver,
                       plot_all_time_steps=False):
    """
    Visualize the solution of the optimization problem
    @param x: State trajectory
    @param u: Input trajectory
    @param x0: Initial state
    @param K: Time horizon
    @param spec_name: Name of the specification
    @param plot_scenario_fct: Function that plots the scenario
    @param system: The system object
    @param solver: The solver
    @param plot_all_time_steps: Flag to plot all time steps
    """
    output_directory = f'outputs/{spec_name}/'
    os.makedirs(output_directory, exist_ok=True)

    # Plot scenario
    if not isinstance(system, DoubleIntegrator) and not isinstance(system, SingleIntegrator):
        fig, axs = plt.subplots(figsize=(10, 10))
        for k in range(K + 1):
            plot_scenario_fct(axs, k)
        plt.plot(*x[:2, :], '-o', color='red', linewidth=3, label='Optimal trajectory')
        plt.plot(x0[0], x0[1], 'o', color='green', linewidth=3, markersize=10, label='Initial state')
        axs.legend()
        axs.set_xlabel('p_x [m]')
        axs.set_ylabel('p_y [m]')
        plt.title('phi := ' + spec_name)
        fig.tight_layout()
        plt.savefig(output_directory + f'scenario_{solver}.png')
        plt.close()

        # Plot scenario for all time steps
        if plot_all_time_steps:
            output_directory_scenario = output_directory + f'scenario_{solver}/'
            os.makedirs(output_directory_scenario, exist_ok=True)

            for k in range(K + 1):
                fig, axs = plt.subplots(figsize=(10, 10))
                plot_scenario_fct(axs, k)
                plt.plot(*x[:2, :k + 1], '-o', color='red', linewidth=3, label='Optimal trajectory')
                plt.plot(x0[0], x0[1], 'o', color='green', linewidth=3, markersize=10, label='Initial state')
                axs.legend()
                axs.set_xlabel('p_x [m]')
                axs.set_ylabel('p_y [m]')
                plt.title('phi := ' + spec_name)

                fig.tight_layout()
                plt.savefig(output_directory_scenario + f'scenario_{solver}_{k}.svg')
                plt.close()


    if isinstance(system, SingleIntegrator):
        # p-t plot
        time_steps = np.arange(K + 1)
        fig, axs = plt.subplots()
        plt.plot(time_steps, x[0, :], '-o', label='pos')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.set_ylim([0, 6])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('pos [m]')
        fig.tight_layout()
        plt.savefig(output_directory + f'pos-t_{solver}.png')
        plt.close()

        # v-t plot
        fig, axs = plt.subplots()
        axs.step(time_steps, u[0, :], '-o', where='post', label='v')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.set_ylim([-10, 10])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('v [m/s]')
        fig.tight_layout()
        plt.savefig(output_directory + f'v-t_{solver}.png')
        plt.close()

    if isinstance(system, DoubleIntegrator):
        # p-t plot
        time_steps = np.arange(K + 1)
        fig, axs = plt.subplots()
        plt.plot(time_steps, x[0, :], '-o', label='pos')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.set_ylim([0, 6])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('pos [m]')
        fig.tight_layout()
        plt.savefig(output_directory + f'pos-t_{solver}.png')
        plt.close()

        # v-t plot
        fig, axs = plt.subplots()
        axs.step(time_steps, x[1, :], '-o', where='post', label='v')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.set_ylim([-10, 10])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('v [m/s]')
        fig.tight_layout()
        plt.savefig(output_directory + f'v-t_{solver}.png')
        plt.close()

        # a-t plot
        fig, axs = plt.subplots()
        axs.step(time_steps, u[0, :], '-o', where='post', label='a')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('a [m/s²]')
        fig.tight_layout()
        plt.savefig(output_directory + f'a-t_{solver}.png')
        plt.close()

    if isinstance(system, PointMass):
        # v-t plot
        time_steps = np.arange(K + 1)
        fig, axs = plt.subplots(figsize=(20, 10))
        plt.plot(time_steps, x[2, :], '-o', label='v_x')
        plt.plot(time_steps, x[3, :], '-o', label='v_y')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('v [m/s]')
        fig.tight_layout()
        plt.savefig(output_directory + f'v-t_{solver}.png')
        plt.close()

        # a-t plot
        fig, axs = plt.subplots(figsize=(20, 10))
        axs.step(time_steps, u[0, :], '-o', where='post', label='a_x')
        axs.step(time_steps, u[1, :], '-o', where='post', label='a_y')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('a [m/s²]')
        fig.tight_layout()
        plt.savefig(output_directory + f'a-t_{solver}.png')
        plt.close()

    if isinstance(system, Unicycle):
        # v-t plot
        time_steps = np.arange(K + 1)
        fig, axs = plt.subplots(figsize=(20, 10))
        plt.plot(time_steps, u[0, :], '-o', label='v_x')
        plt.plot(time_steps, u[1, :], '-o', label='v_y')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('v [m/s]')
        fig.tight_layout()
        plt.savefig(output_directory + f'v-t_{solver}.png')
        plt.close()

    if isinstance(system, Bicycle):
        # delta-v-psi-t plot
        time_steps = np.arange(K + 1)
        fig, axs = plt.subplots(figsize=(20, 10))
        plt.plot(time_steps, x[2, :], '-o', label='delta')
        plt.plot(time_steps, x[3, :], '-o', label='v')
        plt.plot(time_steps, x[4, :], '-o', label='psi')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('delta [rad], v [m/s], psi [rad]')
        fig.tight_layout()
        plt.savefig(output_directory + f'delta-v-psi-t_{solver}.png')
        plt.close()

        # v_delta-a_long-t plot
        fig, axs = plt.subplots(figsize=(20, 10))
        axs.step(time_steps, u[0, :], '-o', where='post', label='v_delta')
        axs.step(time_steps, u[1, :], '-o', where='post', label='a_long')
        axs.legend()
        axs.set_xlim([0, K + 1])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('time step')
        axs.set_ylabel('v_delta [rad/s], a [m/s²]')
        fig.tight_layout()
        plt.savefig(output_directory + f'v_delta-a_long-t_{solver}.png')
        plt.close()

def visualize_pi_steps(spec_name, K, plot_scenario_fct, x0, n_samples, record_y, record_y_opt,
                       record_best_sample_idx):
    """
    Visualizes the iterations of the solution from the PI solver
    @param spec_name: Name of the specification
    @param K: Time horizon
    @param plot_scenario_fct: Function to plot the scenario
    @param x0: Initial time step
    @param n_samples: Number of used samples
    @param record_y: The recorded output trajectories, aka samples
    @param record_y_opt: The PI-optimal output trajectory per iteration
    @param record_best_sample_idx: The cost-optimal output trajectory per iteration
    """
    output_directory = f'outputs/{spec_name}/pi_steps/'

    # Remove the directory if it exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern, similar to LaTeX
    plt.rcParams['font.family'] = 'serif'

    print(f"Number of PI steps recorded: {len(record_y)}")

    for pi_step in range(len(record_y)):
        fig, axs = plt.subplots(figsize=(10, 10))
        for k in range(K + 1):
            plot_scenario_fct(axs, k)

        # All samples
        for sample in range(n_samples):
            plt.plot(*record_y[pi_step][sample][:2, :], '-o', linewidth=1.0, markersize=1.5, color='black', alpha=0.4)

        # Best sample
        best_sample_idx = record_best_sample_idx[pi_step]
        plt.plot(*record_y[pi_step][best_sample_idx][:2, :], '-o', color='skyblue', linewidth=3, label='Best sample')

        # pi results
        plt.plot(*record_y_opt[pi_step][:2, :], '-o', color='red', linewidth=3, label='PI trajectory')

        # Initial state
        plt.plot(x0[0], x0[1], 'o', color='green', linewidth=3, markersize=10, label='Initial state')

        plt.text(0.02, 0.95, f'Iteration: {pi_step}', size=16, color='k', transform=axs.transAxes)

        axs.legend(loc='upper right')

        axs.set_xlabel('p_x [m]')
        axs.set_ylabel('p_y [m]')

        plt.title('phi := ' + spec_name)

        fig.tight_layout()
        plt.savefig(output_directory + 'scenario_pi_step_{:04d}'.format(pi_step) + '.png', format='png', dpi=150)
        plt.close()

    # Save optimal solution
    fig, axs = plt.subplots(figsize=(10,10))
    for k in range(K + 1):
        plot_scenario_fct(axs, k)
    plt.plot(*record_y_opt[-1][:2, :], '-o', color='red', linewidth=3, label='PI trajectory')
    plt.plot(x0[0], x0[1], 'o', color='green', linewidth=3, markersize=10, label='Initial state')

    fig.tight_layout()
    plt.savefig(output_directory + 'scenario_pi_step_solution' + '.png', format='png', dpi=150)
    plt.close()

    # Save initial state scenario
    fig, axs = plt.subplots(figsize=(10, 10))
    for k in range(K + 1):
        plot_scenario_fct(axs, k)

    plt.plot(x0[0], x0[1], 'o', color='green', linewidth=3, markersize=10, label='Initial state')
    fig.tight_layout()
    plt.savefig(output_directory + 'scenario_pi_step_initial' + '.png', format='png', dpi=150)
    plt.close()

    # Create Video
    create_video = False
    if create_video:
        frame_rate = 3
        os.system("ffmpeg -y -r " + str(
            frame_rate) + " -f image2 -i " + output_directory + "scenario_pi_step_%04d.png " + output_directory + "/pi_steps.mp4")


def visualize_pi_steps_single_integrator(spec_name, K, plot_scenario_fct, x0, n_samples, record_y, record_y_opt, record_best_sample_idx):
    """
    Visualizes the iterations of the solution from the PI solver for the single integrator system
    @param spec_name: Name of the specification
    @param K: Time horizon
    @param plot_scenario_fct: Function to plot the scenario
    @param x0: Initial state
    @param n_samples: Number of used samples
    @param record_y: The recorded output trajectories, aka samples
    @param record_y_opt: The PI-optimal output trajectory per iteration
    @param record_best_sample_idx: The cost-optimal output trajectory per iteration
    """
    output_directory = f'outputs/{spec_name}/pi_steps/'

    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern, similar to LaTeX
    plt.rcParams['font.family'] = 'serif'

    # Remove the directory if it exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    for pi_step in range(len(record_y)):

        fig, axs = plt.subplots(figsize=(10, 6))
        time_steps = np.arange(K + 1)

        # All samples
        for sample in range(n_samples):
            plt.plot(time_steps, record_y[pi_step][sample][0, :], '-o', linewidth=0.5, markersize=0.5, color='black', alpha=0.4)

        # Best sample
        best_sample_idx = record_best_sample_idx[pi_step]
        plt.plot(time_steps, record_y[pi_step][best_sample_idx][0, :], '-o', color='skyblue', linewidth=3, label='Best sample')

        # PI results
        plt.plot(time_steps, record_y_opt[pi_step][0, :], '-o', color='red', linewidth=3, label='PI trajectory')

        plt.text(0.05, 0.90, f'Iteration: {pi_step}', size=15, color='Gray', transform=axs.transAxes)
        axs.legend(loc='upper right')

        plt.title('phi := ' + spec_name)

        axs.set_xlim([0, K])
        axs.set_ylim([0, 6])
        axs.grid(which='minor', alpha=0.2)
        axs.grid(which='major', alpha=0.5)
        axs.set_xlabel('k')
        axs.set_ylabel('x')
        fig.tight_layout()

        plt.savefig(output_directory + 'scenario_pi_step_{:04d}'.format(pi_step) + '.png', format='png', dpi=100)
        plt.close()


    # Save optimal solution
    fig, axs = plt.subplots(figsize=(10, 6))
    time_steps = np.arange(K + 1)
    plt.plot(time_steps, record_y_opt[-1][0, :], '-o', color='red', linewidth=3, label='PI trajectory')
    axs.set_xlim([0, K])
    axs.set_ylim([0, 6])
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel('k')
    axs.set_ylabel('x')
    fig.tight_layout()
    plt.savefig(output_directory + 'scenario_pi_step_solution' + '.png', format='png', dpi=100)
    plt.close()

    # Save initial state scenario
    fig, axs = plt.subplots(figsize=(10, 6))
    time_steps = np.arange(K + 1)
    axs.set_xlim([0, K])
    axs.set_ylim([0, 6])
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel('k')
    axs.set_ylabel('x')
    fig.tight_layout()
    plt.savefig(output_directory + 'scenario_pi_step_initial' + '.png', format='png', dpi=100)
    plt.close()

    fig, axs = plt.subplots()
    time_steps = np.arange(K + 1)

    # Define the timesteps to be plotted and corresponding colors
    timesteps_to_plot = [0, 3, 6, 9]
    colors = ['blue', 'green', 'orange', 'purple']
    legend_entries = []

    if len(record_y) == 0:
        print('No trajectory samples were recorded. Probably the c++ implementation was used, where this is turned off by '
              'default. To plot the samples, use the python PI implementation.')

    for idx, pi_step in enumerate(timesteps_to_plot):

        # All samples
        if len(record_y) > 0:
            for sample in range(0, n_samples, 2):
                plt.plot(time_steps, record_y[pi_step][sample][0, :], '-o', linewidth=0.5, markersize=0.0, color=colors[idx], alpha=0.2)
            # Adding a label for the first sample to create a legend entry
            plt.plot(time_steps, record_y[pi_step][0][0, :], '-o', linewidth=0.5, markersize=0.5, color=colors[idx], alpha=0.4, label=f'Samples at k={pi_step}')

        # Best sample
        #best_sample_idx = record_best_sample_idx[pi_step]
        #plt.plot(time_steps, record_y[pi_step][best_sample_idx][0, :], '-o', color=colors[idx], linewidth=3, label=f'Best sample (step {pi_step})')

        legend_entries.append(f'Step {pi_step} samples')

    # PI results
    plt.plot(time_steps, record_y_opt[pi_step][0, :], '-o', color='red', linewidth=3, label='PI trajectory')

    axs.legend(loc='upper right')

    #plt.title('phi := ' + spec_name)

    axs.set_xlim([0, K ])
    axs.set_ylim([0, 6])
    axs.grid(which='minor', alpha=0.2)
    axs.grid(which='major', alpha=0.5)
    axs.set_xlabel('time step')
    axs.set_ylabel('pos [m]')
    fig.tight_layout()

    plt.savefig(output_directory + 'scenario_pi_steps_combined.svg', dpi=200)
    plt.close()

def visualize_solution_comparison(solutions, K, name, plot_scenario_fct):
    """
    Visualizes the solutions of all solvers
    @param solutions: The solutions for each solver
    @param K: Time horizon
    @param name: Name
    @param plot_scenario_fct: Function to plot the scenario
    """
    output_directory = f'outputs/{name}/'
    os.makedirs(output_directory, exist_ok=True)

    # Plot scenario
    fig, axs = plt.subplots(figsize=(10, 10))
    for k in range(K + 1):
        plot_scenario_fct(axs, k)

    for solver, solution in solutions.items():
        plt.plot(*solution.x[:2, :], '-o', linewidth=3, label=solver + f', rho: {round(solution.rho, 3)}')
    axs.legend()
    axs.set_xlabel('p_x [m]')
    axs.set_ylabel('p_y [m]')
    plt.title('phi := ' + name)
    fig.tight_layout()
    plt.savefig(output_directory + f'compare_solvers.svg')
    plt.close()


def save_table(table, spec_name):
    """
    Saves a comparison of the solutions of the solvers as sable
    @param table: Table data
    @param spec_name: Name of the specification
    """
    output_directory = f'outputs/{spec_name}/'
    os.makedirs(output_directory, exist_ok=True)

    with open(output_directory + 'solver_comparison.txt', 'w') as file:
        file.write(table)
