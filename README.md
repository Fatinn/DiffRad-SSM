

# 📡 DifferentiableDSP for mmWave Radar Signal Processing

This repository provides an **end-to-end differentiable Digital Signal Processing (DSP) pipeline** for processing raw ADC data from **TI DCA1000 + mmWave radar sensors**.
The module converts raw complex radar signals into **Range–Doppler (RD) maps** using fully differentiable FFT and learnable window functions, making it suitable for deep learning models such as Radar-MambaNet.

---

## 📌 Features

* ✅ Read raw ADC data from `adc_data.bin`
* ✅ Automatically pad or trim ADC data to match configuration
* ✅ Learnable or fixed Hamming windows
* ✅ Fully differentiable Range FFT + Doppler FFT
* ✅ Log-magnitude RD map output
* ✅ Ready to integrate with PyTorch deep networks
* ✅ Built-in visualization example

---

## 📁 Project Structure

```
.
├── DifferentiableDSP.py        # Main processing code
├── adc_data.bin                # Radar raw ADC file (optional)
└── README.md
```

---

## 🛠️ Installation

```bash
pip install torch numpy matplotlib
```

No extra dependencies are required.

---

## ⚙️ Radar Configuration

`RadarConfig` contains all radar acquisition parameters:

```python
class RadarConfig:
    START_FREQ = 77.0         # GHz
    FREQ_SLOPE = 29.982       # MHz/us
    IDLE_TIME = 40            # us
    RAMP_END_TIME = 60        # us
    ADC_START_TIME = 6        # us
    SAMPLE_RATE = 6000        # ksps

    NUM_SAMPLES = 512
    NUM_RX = 4
    NUM_CHIRPS = 160
    NUM_FRAMES = 40

    FILE_PATH = "adc_data.bin"
```

These parameters must match your radar configuration (e.g., TI IWR6843, AWR1843).

---

## 📥 Reading Raw ADC Data

Raw DCA1000 data is loaded using:

```python
raw_data_numpy = read_dca1000_bin(cfg.FILE_PATH, cfg)
```

### If `adc_data.bin` does not exist:

A random complex tensor simulating radar data will be generated:

```
shape = [Frames, Chirps, RX, Samples]
       = [40, 160, 4, 512]
```

### If real DCA1000 data exists:

* Reads 16-bit signed integers
* Converts I/Q pair into complex values
* Reshapes into 4-D radar data cube

---

## 🔧 Differentiable DSP Module

```python
dsp_layer = DifferentiableDSP(
    num_samples=cfg.NUM_SAMPLES,
    num_chirps=cfg.NUM_CHIRPS,
    learnable=True
)
```

### Pipeline Steps

| Step | Operation            | Differentiable |
| ---- | -------------------- | -------------- |
| 1    | Apply range window   | Yes            |
| 2    | Range FFT            | Yes            |
| 3    | Apply Doppler window | Yes            |
| 4    | Doppler FFT          | Yes            |
| 5    | FFT shift            | Yes            |
| 6    | Log magnitude (dB)   | Yes            |

The layer outputs:

```
[Batch, Frames, RX, Doppler, Range]
```

Example:

```
[1, 40, 4, 160, 512]
```

---

## ▶️ Running the Script

```bash
python DifferentiableDSP.py
```

Console output example:

```
--- Radar-MambaNet Preprocessing ---
Parameters: Start=77.0GHz, Samples=512, Slope=29.982MHz/us
Running on device: cuda
[Success] Output Tensor Shape: torch.Size([1, 40, 4, 160, 512])
Format: [Batch, Frames, RX_Channels, Doppler, Range]
```

---

## 🎨 Visualization Example

The script automatically visualizes the RD map of:

```
Frame 0, RX antenna 0
```

Example code:

```python
plt.imshow(vis_data, aspect='auto', cmap='jet', origin='lower')
plt.title("Differentiable RD Map (Frame 0, RX 0)")
plt.xlabel("Range Bins (512)")
plt.ylabel("Doppler Bins")
plt.colorbar(label="Magnitude (dB)")
plt.show()
```

Result example:

* X-axis: Range bins
* Y-axis: Doppler bins
* Color: dB magnitude

---

## 🤖 Integrating with a Deep Learning Model

You can directly insert this module at the front of your neural pipeline:

```python
dsp = DifferentiableDSP(512, 160, learnable=True)

rd_map = dsp(raw_adc_tensor)      # Output suitable for CNNs, Transformers, Mamba, etc.
```

This makes your entire radar perception pipeline:

* End-to-end trainable
* Able to optimize DSP windowing
* Avoid offline preprocessing

---

## 📌 Output Format Summary

| Dimension | Meaning           |
| --------- | ----------------- |
| Batch     | Dataset batch     |
| Frames    | Radar frames      |
| RX        | Receiver channels |
| Doppler   | Doppler FFT bins  |
| Range     | Range FFT bins    |

Final output example:

```
Shape: [1, 40, 4, 160, 512]
```

---

## 📄 License

This project is released under the MIT License.

---

## ✉️ Contact

For questions or improvements, feel free to open an issue or contact the author.

---

