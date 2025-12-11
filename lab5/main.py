import soundfile as sf
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt


def noise_reduction(input_file_path, output_file_path, frame_size=4096, overlap=0.5, noise_start=0, noise_end=2000, suppression_factor=1.3, protection_factor=0.0001):
    audio_data, sample_rate = sf.read(input_file_path)

    # Преобразование аудио в моно
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis = 1)

    # Нормализация
    audio_data = audio_data.astype(np.float32)
    peak = np.max(np.abs(audio_data))
    audio_data /= peak

    hop_size = int(frame_size * (1-overlap))
    window = get_window('hann', frame_size)

    noise_frames = []
    noise_start_in_samples = noise_start * sample_rate // 1000
    noise_end_in_samples = min(noise_end * sample_rate // 1000, len(audio_data) - frame_size)
    for i in range(noise_start_in_samples, noise_end_in_samples, hop_size):
        frame = audio_data[i:i + frame_size] * window
        noise_fft = np.fft.fft(frame)
        noise_frames.append(np.abs(noise_fft))
    
    noise_profile = np.mean(noise_frames, axis=0) if noise_frames else np.zeros(frame_size)

    output_audio_signal = np.zeros(len(audio_data))
    window_sum = np.zeros(len(audio_data))

    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i+frame_size] * window
        frame_fft = np.fft.fft(frame)

        magnitude = np.abs(frame_fft) # амплитуда на каждой частоте
        phase = np.angle(frame_fft) # фаза

        clean_magnitude = np.maximum(magnitude - suppression_factor * noise_profile, protection_factor * magnitude)

        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_frame = np.real(np.fft.ifft(clean_fft))

        output_audio_signal[i:i+frame_size] += clean_frame
        window_sum[i:i+frame_size] += window
    
    window_sum[window_sum==0] = 1
    output_audio_signal /= window_sum

    # приведение к нужному формату звука
    output_audio_signal = output_audio_signal * 32767
    output_audio_signal = np.clip(output_audio_signal, -32768, 32767)
    output_audio_signal = output_audio_signal.astype(np.int16)

    sf.write(output_file_path, output_audio_signal, sample_rate)

    # спектрограммы
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.specgram(audio_data, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Спектрограмма исходного сигнала')
    plt.ylabel('Частота')
    plt.xlabel('Время (секунды)')

    plt.subplot(2, 1, 2)
    plt.specgram(output_audio_signal, Fs=sample_rate, NFFT=frame_size, noverlap=hop_size)
    plt.colorbar()
    plt.title('Спектрограмма обработанного сигнала')
    plt.ylabel('Частота')
    plt.xlabel('Время (секунды)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    noise_reduction('./sounds/dieselengine.wav', './sounds/dieselengine_noise_reduced.wav')
