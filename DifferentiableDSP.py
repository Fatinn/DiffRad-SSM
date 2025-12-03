import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

class RadarConfig:
    START_FREQ = 77.0         
    FREQ_SLOPE = 29.982      
    IDLE_TIME = 40            
    RAMP_END_TIME = 60        
    ADC_START_TIME = 6         
    SAMPLE_RATE = 6000        
    

    NUM_SAMPLES = 512          
    NUM_RX = 4                
    

    NUM_CHIRPS = 160          
    NUM_FRAMES = 40            
    
    # 数据路径
    FILE_PATH = "adc_data.bin"


def read_dca1000_bin(file_path, config):


    if not os.path.exists(file_path):


        raw_data = np.random.randn(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES) + \
                   1j * np.random.randn(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES)
        return raw_data.astype(np.complex64)


    adc_data = np.fromfile(file_path, dtype=np.int16)
 
    adc_data = adc_data.reshape(-1, 2)
    complex_data = adc_data[:, 0] + 1j * adc_data[:, 1]
    

    
    expected_points = config.NUM_FRAMES * config.NUM_CHIRPS * config.NUM_RX * config.NUM_SAMPLES

    if complex_data.size != expected_points:
        
        if complex_data.size > expected_points:
            complex_data = complex_data[:expected_points]
        else:
            complex_data = np.pad(complex_data, (0, expected_points - complex_data.size))


    raw_cube = complex_data.reshape(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES)
    
    return raw_cube


class DifferentiableDSP(nn.Module):
    def __init__(self, num_samples, num_chirps, learnable=True):
        super(DifferentiableDSP, self).__init__()
        
    
        range_win = torch.hamming_window(num_samples)
        doppler_win = torch.hamming_window(num_chirps)
        
        if learnable:
            self.range_window = nn.Parameter(range_win)
            self.doppler_window = nn.Parameter(doppler_win)
        else:
            self.register_buffer('range_window', range_win)
            self.register_buffer('doppler_window', doppler_win)

    def forward(self, x):
        """
        x: [Batch, Frames, Chirps, RX, Samples] 
        out:   [Batch, Frames, RX, Doppler, Range] (Log-Magnitude)
        """
        

        x_w = x * self.range_window
   
        range_fft = torch.fft.fft(x_w, dim=-1)
        

        d_win = self.doppler_window.view(1, 1, -1, 1, 1)
        range_fft_w = range_fft * d_win
        

        doppler_fft = torch.fft.fft(range_fft_w, dim=2)

        doppler_fft = torch.fft.fftshift(doppler_fft, dim=2)
        

        mag = torch.abs(doppler_fft)

        out = 20 * torch.log10(mag + 1e-5)
        

        out = out.permute(0, 1, 3, 2, 4)
        
        return out



if __name__ == "__main__":
    cfg = RadarConfig()
    print(f"--- Radar-MambaNet Preprocessing ---")
    print(f"Parameters: Start={cfg.START_FREQ}GHz, Samples={cfg.NUM_SAMPLES}, Slope={cfg.FREQ_SLOPE}MHz/us")


    raw_data_numpy = read_dca1000_bin(cfg.FILE_PATH, cfg)
    

    raw_tensor = torch.from_numpy(raw_data_numpy).unsqueeze(0)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    raw_tensor = raw_tensor.to(device)
    

    dsp_layer = DifferentiableDSP(cfg.NUM_SAMPLES, cfg.NUM_CHIRPS, learnable=True).to(device)
    

    with torch.no_grad(): 
        rd_maps = dsp_layer(raw_tensor)
    
    print(f"\n[Success] Output Tensor Shape: {rd_maps.shape}")
    print(f"Format: [Batch, Frames, RX_Channels, Doppler, Range]")
    print(f" [1, {cfg.NUM_FRAMES}, {cfg.NUM_RX}, {cfg.NUM_CHIRPS}, {cfg.NUM_SAMPLES}]")


    if device.type == 'cuda':
        vis_data = rd_maps[0, 0, 0, :, :].cpu().numpy()
    else:
        vis_data = rd_maps[0, 0, 0, :, :].numpy()
        
    plt.figure(figsize=(10, 6))

    plt.imshow(vis_data, aspect='auto', cmap='jet', origin='lower')
    plt.title(f"Differentiable RD Map (Frame 0, RX 0)\nSamples={cfg.NUM_SAMPLES}, Learnable Window")
    plt.xlabel("Range Bins (512)")
    plt.ylabel("Doppler Bins")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.show()
    