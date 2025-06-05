# fusion-simulator-suite
    A Python-based desktop application for analyzing and visualizing fusion reactor concepts using realistic physics models and comparative presets, using Fibonacci spiral coil format
# Fusion Simulation Suite

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project is a desktop application for analyzing and visualizing fusion reactor concepts. It was built to critically evaluate conceptual designs by grounding them in realistic plasma physics, inspired by the expert analysis in the "Whitepaper Simulation Validation Plan."

The application provides two core tools in a tabbed interface: one for rigorous, physics-based feasibility analysis and another for simplified, illustrative particle visualization.

## Key Features

-   **Dual-Tool Interface:** A tabbed GUI separating the "Reactor Feasibility Analysis" from the "Particle Visualizer."
-   **Realistic Physics Engine:** The feasibility tool uses the Bosch-Hale model for D-D fusion reactivity and calculates a full power balance, including Bremsstrahlung radiation and transport losses.
-   **Comparative Presets:** Instantly load and compare the projected parameters of real-world designs like **ITER** and **SPARC** against conceptual ones.
-   **Performance Optimized:** The visualizer uses multiprocessing to run particle simulations in parallel, significantly speeding up performance on multi-core CPUs.
-   **Rich User Feedback:** Includes a real-time progress bar and a "time remaining" estimate for complex simulations.
-   **Advanced Visualization:** Generates 3D trajectory plots, energy-over-time graphs, and creates GIF animations of particle paths, complete with a "predicted future path" overlay.
-   **Comprehensive Reporting:** Exports feasibility analysis reports in **PDF**, **HTML**, and **CSV** formats.

---

### The Two Core Tools

#### 1. Reactor Feasibility Analysis

This tool is designed to answer the fundamental question: "Is this fusion concept energetically viable?" It uses a 0D power balance model to calculate a realistic Q-Value based on key plasma and machine parameters. It serves as a scientific tool to test if a reactor concept can produce net energy under the laws of fusion physics.

#### 2. Particle Visualizer

This tool provides a simplified, visual representation of how particles behave in a magnetic field. It replicates the look and feel of a more basic simulator, with features including:
-   3D trajectory plotting.
-   Kinetic energy vs. timestep graphs.
-   An animation engine that saves GIFs and displays the animation in a new window.

---

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/YourUsername/your-repository-name.git](https://github.com/YourUsername/your-repository-name.git)
    cd your-repository-name
    ```
2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install the required libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

Once the requirements are installed, run the main application file from the project's root directory:

```bash
python main.py
```

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
