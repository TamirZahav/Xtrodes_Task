#Tamir Zahav - ID-207107723
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
import matplotlib.pyplot as plt

# Creating The required class
class SignalProcessor:
    def __init__(self, file_path, num_channels=8, samp_ps=4000, num_bits=15, voltage_res=4.12e-7):
        self.file_path = file_path
        self.num_channels = num_channels
        self.samp_ps = samp_ps
        self.num_bits = num_bits
        self.voltage_res = voltage_res
        self.data = None

    def load_and_convert_data(self):
        # Load data from file, reshape it (switch lines and rows), changing to volt values and insert it to 'table'
        self.data = np.fromfile(self.file_path, dtype=np.uint16)
        self.data = np.reshape(self.data, (self.num_channels, -1), order='F')
        self.data = np.multiply(self.voltage_res, (self.data - np.float_power(2, self.num_bits - 1)))
        self.df = pd.DataFrame(self.data.T, columns=[f'Channel_{i + 1}' for i in range(self.num_channels)])

    def zero_phase_bandpass_filter(self, lowcut=10, highcut=1000, fs=4000, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Create the bandpass filter
        b, a = butter(order, [low, high], btype='band')

        # Apply the filter with zero-phase distortion
        filtered_data = filtfilt(b, a, self.data)
        return filtered_data

def main():
    file_path = '/Users/tamirzahav/Desktop/NEUR0000.DT8'
    signal_processor = SignalProcessor(file_path)
    signal_processor.load_and_convert_data()

    # Transpose unfiltered signal
    unfiltered_signal = signal_processor.data.T
    # Filtering
    filtered_signal = signal_processor.zero_phase_bandpass_filter()

    # Transpose the filtered signal
    filtered_signal = filtered_signal.T

    # Determine the number of samples in one second and 250 seconds
    num_samples_one_sec = signal_processor.samp_ps
    num_samples_250_sec = num_samples_one_sec * 250

    # Slice the data to include only the first second and 250 seconds
    unfiltered_signal_1sec = unfiltered_signal[:num_samples_one_sec, :]
    filtered_signal_1sec = filtered_signal[:num_samples_one_sec, :]

    unfiltered_signal_250sec = unfiltered_signal[:num_samples_250_sec, :]
    filtered_signal_250sec = filtered_signal[:num_samples_250_sec, :]

    # Adjust the time axis for both views
    time_axis_1sec = np.arange(unfiltered_signal_1sec.shape[0]) / signal_processor.samp_ps
    time_axis_250sec = np.arange(unfiltered_signal_250sec.shape[0]) / signal_processor.samp_ps

    # Plotting the unfiltered and filtered signals for each channel
    for channel in range(signal_processor.num_channels):
        plt.figure(figsize=(14, 12))

        # Plot unfiltered signal for 1 second
        plt.subplot(2, 2, 1)
        plt.plot(time_axis_1sec, unfiltered_signal_1sec[:, channel], label=f'Unfiltered Channel {channel + 1}', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Unfiltered Signal - 1 Sec - Channel {channel + 1}')
        plt.legend()
        plt.grid(True)

        # Plot filtered signal for 1 second
        plt.subplot(2, 2, 2)
        plt.plot(time_axis_1sec, filtered_signal_1sec[:, channel], label=f'Filtered Channel {channel + 1}', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Filtered Signal - 1 Sec - Channel {channel + 1}')
        plt.legend()
        plt.grid(True)

        # Plot unfiltered signal for 250 seconds
        plt.subplot(2, 2, 3)
        plt.plot(time_axis_250sec, unfiltered_signal_250sec[:, channel], label=f'Unfiltered Channel {channel + 1}', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Unfiltered Signal - 250 Sec - Channel {channel + 1}')
        plt.legend()
        plt.grid(True)

        # Plot filtered signal for 250 seconds
        plt.subplot(2, 2, 4)
        plt.plot(time_axis_250sec, filtered_signal_250sec[:, channel], label=f'Filtered Channel {channel + 1}', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Filtered Signal - 250 Sec - Channel {channel + 1}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
