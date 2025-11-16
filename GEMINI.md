# Gemini Project: Advanced C++ Accelerated Calculus Trading System

## Project Overview

This project is a sophisticated, high-performance algorithmic trading system designed for the cryptocurrency market, specifically interacting with the Bybit exchange. It implements a calculus-based trading strategy, using derivatives of price (velocity and acceleration) to generate trading signals.

The system's architecture is a hybrid of Python and C++, designed to leverage the strengths of both languages. Python is used for the high-level trading logic, orchestration, and API communication, while C++ is employed for performance-critical mathematical computations, providing a significant speedup (reportedly over 10x) for tasks like Kalman filtering and curve analysis.

### Core Components:

*   **`live_calculus_trader.py`**: The main entry point for running the live or simulated trading bot.
*   **`calculus_strategy.py`**: Contains the core logic for generating trading signals based on price velocity and acceleration.
*   **`risk_manager.py` & `portfolio_manager.py`**: Handle position sizing, risk controls (e.g., stop-loss, daily limits), and multi-asset portfolio optimization.
*   **`bybit_client.py` & `websocket_client.py`**: Manage communication with the Bybit exchange for market data and order execution.
*   **`cpp/` directory**: Contains the C++ source code for accelerated mathematical functions (e.g., Kalman filters, curve analysis, risk calculations).
*   **`cpp_bridge_working.py`**: The Python interface (using pybind11) to the compiled C++ library, allowing Python code to call the high-performance C++ functions.
*   **`config.py`**: Central configuration file for API keys, trading parameters, and risk settings.

## Building and Running

### 1. Prerequisites

*   Python 3.8+
*   A C++ compiler (e.g., `build-essential` on Linux, `Xcode Command Line Tools` on macOS)
*   Bybit API credentials

### 2. Install Python Dependencies

The project uses several Python libraries. Install them using pip:

```bash
pip install numpy pandas scipy PyWavelets
```

### 3. Build the C++ Core

The C++ components must be compiled into a shared library that Python can use. A build script is provided for this purpose.

```bash
# Navigate to the scripts directory
cd scripts

# Execute the build script
./build_working.sh
```

This will compile the C++ source in the `cpp/` directory and place the output in the `cpp_bridge/` directory, making it accessible to the Python code.

### 4. Configure the System

Before running, you must add your Bybit API key and secret to the configuration file.

```bash
# Edit the configuration file
vim config.py
```

Update the `BYBIT_API_KEY` and `BYBIT_API_SECRET` fields. You can also adjust trading parameters like leverage, risk, and loss limits in this file.

### 5. Running the Trader

The main application is `live_calculus_trader.py`.

**To run in simulation mode (recommended for testing):**
This mode connects to the live market data but does not execute real trades. It's useful for verifying that the signal generation and risk management are working as expected.

```bash
python3 live_calculus_trader.py --simulation
```

**To run in live trading mode:**
**WARNING: This will execute real trades with real money.** Use with caution.

```bash
python3 live_calculus_trader.py
```

## Development Conventions

*   **Modularity**: The codebase is organized into distinct modules with clear responsibilities (e.g., `risk_manager`, `calculus_strategy`).
*   **Performance-First**: For computationally intensive tasks, the logic is offloaded to C++ functions. When adding new mathematical models, consider whether a C++ implementation is necessary for performance.
*   **Fail-Safes**: The system includes numerous safety features, such as daily loss limits, emergency stops, and graceful degradation (falling back to Python implementations if C++ components fail). New features should integrate with these safety mechanisms.
*   **Configuration-Driven**: Hardcoded values are discouraged. Key parameters for trading strategies and risk management should be defined in `config.py`.
*   **Logging**: The system provides detailed logs for all major events, including signal generation, order placement, and errors. Use the logging infrastructure to provide visibility into new features.