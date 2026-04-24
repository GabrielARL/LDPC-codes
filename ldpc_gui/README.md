# LDPC GUI Application

A standalone GUI application for LDPC (Low-Density Parity-Check) code simulation and decoding using Belief Propagation algorithm.

## Features

- Interactive parameter selection for LDPC codes
- ISI (Inter-Symbol Interference) channel simulation
- Belief Propagation decoding
- C++ DFEC dataset decoder tab for exported `cpp_bundle` data
- Packet-by-packet DFEC progress with live packet success rate
- Selectable pilot-ratio presets
- Real-time pipeline execution status
- Button-based interface for easy operation

## Building

```bash
cd ldpc_gui
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./ldpc_gui
```

From the repository root, you can also use the launcher that starts the GUI
from the correct working directory for the decoder pipeline:

```bash
./run_bp_decoder_ui.sh
```

On Linux desktops, the repository root also includes a double-clickable
launcher file: `BP Decoder UI.desktop`.

## Usage

1. Select LDPC code parameters (block length, rate, etc.)
2. Configure channel parameters (SNR, ISI taps)
3. Click "Run Pipeline" to execute the simulation
4. Monitor progress and results in real-time

For the C++ DFEC dataset flow:

1. Open the `DFEC Dataset` tab
2. Paste or drag-and-drop a `DFEC_experiment/export/cpp_bundle` directory (or its `manifest.txt`)
3. Choose one or more pilot-ratio presets
4. Click `Run C++ DFEC Decode`
5. Watch packets decode one by one, with packet success defined as `DFEC BER = 0`

## Dependencies

- GLFW 3.4
- Dear ImGui 1.91.8
- OpenGL
- X11 (Linux)

The application automatically downloads and builds required dependencies during the CMake configuration.
