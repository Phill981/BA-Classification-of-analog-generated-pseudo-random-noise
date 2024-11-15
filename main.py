import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate, resample
from scipy.stats import entropy
from scipy.fft import fft
from joblib import Parallel, delayed
from memory_profiler import profile
import gc

sns.set(style="whitegrid", palette="muted")

class NoiseAnalysis:
    def __init__(self, signal_csv, compare_signal_csv=None, noise_colors=None):
        self.signal_csv = signal_csv
        self.compare_signal_csv = compare_signal_csv
        self.noise_colors = noise_colors if noise_colors else {
            'white': self.white_noise_psd(),
            'pink': self.pink_noise_psd(),
            'brown': self.brown_noise_psd(),
            'blue': self.blue_noise_psd(),
            'violet': self.violet_noise_psd()
        }
        
    def read_signal(self, csv_file):
        """Reads signal data from a CSV file in chunks to avoid memory overload."""
        data = pd.read_csv(csv_file, sep=";", usecols=['# TIME (ns)', 'CH1 (V)'], 
                           dtype={'# TIME (ns)': np.float32, 'CH1 (V)': np.float32})
        return data['CH1 (V)'].values
    
    def white_noise_psd(self):
        return np.ones(1000)
    
    def pink_noise_psd(self):
        freqs = np.fft.fftfreq(1000)
        psd = 1 / np.abs(freqs)
        psd[0] = 0
        return psd

    def brown_noise_psd(self):
        freqs = np.fft.fftfreq(1000)
        freqs = freqs[freqs > 0]
        psd = 1 / (freqs ** 2)
        psd = np.concatenate(([0], psd))
        psd = psd / np.sum(psd)
        return psd

    def blue_noise_psd(self):
        freqs = np.fft.fftfreq(1000)
        psd = np.abs(freqs)
        psd[0] = 0
        return psd

    def violet_noise_psd(self):
        freqs = np.fft.fftfreq(1000)
        psd = (np.abs(freqs) ** 2)
        psd[0] = 0
        return psd

    def fast_fourier_transform(self, signal):
        N = len(signal)
        fft_result = fft(signal)
        psd = np.abs(fft_result[:N // 2])**2
        return psd

    def wavelet_transform(self, signal):
        psd = np.abs(np.fft.fft(signal))**2
        return psd

    def kl_divergence(self, measured_psd, reference_psd):
        epsilon = 1e-10
        measured_psd = np.clip(measured_psd, epsilon, None)
        reference_psd = np.clip(reference_psd, epsilon, None)
        if np.any(np.isnan(measured_psd)) or np.any(np.isnan(reference_psd)) or \
           np.any(np.isinf(measured_psd)) or np.any(np.isinf(reference_psd)):
            return np.nan
        return entropy(measured_psd, reference_psd)

    def plot_psd(self, signal_psd, noise_colors_psd, method, file_name):
        plt.figure(figsize=(10, 6))
        plt.plot(signal_psd, label="Signal PSD", color="blue", linewidth=2)
        for noise_color, reference_psd in noise_colors_psd.items():
            plt.plot(reference_psd, label=f"{noise_color} Noise", linewidth=2)
        plt.yscale('log')
        plt.title(f'PSD Comparison with Noise Models - {method}', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Power', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/graphs/PSD-Comparison-{file_name}-{method}.png")
        plt.close()

    def compare_with_known_noise(self, method='fourier', file_name='signal'):
        signal = self.read_signal(self.signal_csv)
        
        if method == 'fourier':
            signal_psd = self.fast_fourier_transform(signal)
        elif method == 'wavelet':
            signal_psd = self.wavelet_transform(signal)
        else:
            raise ValueError("Invalid method. Choose 'fourier' or 'wavelet'.")

        signal_psd = signal_psd / np.sum(signal_psd)
        signal_psd = np.clip(signal_psd, 1e-10, None)

        noise_colors_psd = {}
        kl_divergences = {}
        
        for noise_color, reference_psd in self.noise_colors.items():
            reference_psd_resampled = resample(reference_psd, len(signal_psd))
            reference_psd_resampled = reference_psd_resampled / np.sum(reference_psd_resampled)
            reference_psd_resampled = np.clip(reference_psd_resampled, 1e-10, None)
            noise_colors_psd[noise_color] = reference_psd_resampled
            kl_div = self.kl_divergence(signal_psd, reference_psd_resampled)
            kl_divergences[noise_color] = kl_div
            print(f'KL Divergence with {noise_color} noise: {kl_div}')

        self.plot_psd(signal_psd, noise_colors_psd, method, file_name)
        return kl_divergences

    def autocorrelation(self, signal):
        """Computes the normalized autocorrelation of the signal."""
        signal = signal - np.mean(signal)
        signal = signal / np.std(signal)
        
        auto_corr = correlate(signal, signal, mode='full')
        auto_corr = auto_corr / np.max(auto_corr)
        
        return auto_corr
    
    def plot_and_save_autocorrelation(self, signal, signal_name, filename="autocorrelation_plot"):
        """Plots and saves the autocorrelation of the signal."""
        # Compute autocorrelation
        auto_corr = self.autocorrelation(signal)
        
        lags = np.arange(-len(signal) + 1, len(signal))
        
        plt.figure(figsize=(10, 6))
        plt.plot(lags, auto_corr, label="Autocorrelation", color="blue", linewidth=2)
        plt.title(f'Autocorrelation of {signal_name}', fontsize=16)
        plt.xlabel('Lag', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs("results/graphs", exist_ok=True)
        
        plot_filename = f"results/graphs/{filename}-{signal_name}-autocorrelation.png"
        plt.savefig(plot_filename)
        plt.close()

    def cross_correlation_matrix(self, signals, signal_names):
        """Computes the cross-correlation matrix for multiple signals."""
        num_signals = len(signals)
        cross_corr_matrix = np.zeros((num_signals, num_signals))
        
        for i in range(num_signals):
            for j in range(i, num_signals):
                signal_i = signals[i] - np.mean(signals[i])
                signal_j = signals[j] - np.mean(signals[j])
                
                signal_i /= np.std(signal_i)
                signal_j /= np.std(signal_j)
                
                cross_corr = correlate(signal_i, signal_j, mode='full')
                
                cross_corr /= len(signal_i)
                
                cross_corr_matrix[i, j] = np.mean(cross_corr)
        
        cross_corr_df = pd.DataFrame(cross_corr_matrix, index=signal_names, columns=signal_names)
        return cross_corr_df

    def process_kl_divergence(self, filenames, data_directory, kl_matrices):
        """Processes KL divergence for each file and updates the KL matrices."""
        for filename in filenames:
            signal_csv = os.path.join(data_directory, filename)
            signal_name = filename.replace(".csv", "")

            noise_analysis = NoiseAnalysis(signal_csv)

            for method in ['fourier', 'wavelet']:
                kl_divergences = noise_analysis.compare_with_known_noise(method=method, file_name=filename)
                for noise_color, kl_value in kl_divergences.items():
                    kl_matrices[method].loc[noise_color, signal_name] = kl_value

            del noise_analysis
            gc.collect()

    def process_cross_correlation(self, filenames, data_directory):
        """Processes cross-correlation for each file and generates the cross-correlation matrix."""
        signals = []
        signal_names = []
        for filename in filenames:
            signal_csv = os.path.join(data_directory, filename)
            signal_name = filename.replace(".csv", "")

            signal = self.read_signal(signal_csv)

            signals.append(signal)
            signal_names.append(signal_name)

        cross_corr_df = self.cross_correlation_matrix(signals, signal_names)
        print("Cross-correlation matrix:")
        print(cross_corr_df)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_corr_df, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'})
        plt.title("Cross-Correlation Matrix")
        plt.tight_layout()
        plt.savefig("results/graphs/cross_correlation_matrix.png")
        plt.close()

    def run(self, data_directory, filenames):
        """Main method to run the entire analysis."""
        kl_matrices = {'fourier': pd.DataFrame(index=self.noise_colors.keys(), columns=filenames),
                    'wavelet': pd.DataFrame(index=self.noise_colors.keys(), columns=filenames)}

        self.process_kl_divergence(filenames, data_directory, kl_matrices)

        kl_matrices['fourier'].to_csv("results/kl_fourier_divergences.csv")
        kl_matrices['wavelet'].to_csv("results/kl_wavelet_divergences.csv")

        self.process_cross_correlation(filenames, data_directory)

        for filename in filenames:
            signal_csv = os.path.join(data_directory, filename)
            signal_name = filename.replace(".csv", "")

            signal = self.read_signal(signal_csv)

            self.plot_and_save_autocorrelation(signal, signal_name, filename="autocorrelation")


if __name__ == "__main__":
    data_directory= "./data"
    
    filenames = [f for f in os.listdir(data_directory) if f.endswith(".csv")]

    noise_analysis = NoiseAnalysis(filenames[0])
    noise_analysis.run(data_directory, filenames)