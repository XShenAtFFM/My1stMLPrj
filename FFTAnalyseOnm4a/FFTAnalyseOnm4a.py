import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment

def spectrum_analysis(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path, format="m4a")
    
    # Export the audio file as wav
    audio.export("temp.wav", format="wav")
    
    # Read the wav file
    sample_rate, data = wavfile.read("temp.wav")
    try:
        os.remove("temp.wav")
    except OSError:
        pass
    
    # Perform Fourier Transform
    n = len(data)
    k = np.arange(n)
    T = n / sample_rate
    frq = k / T
    frq = frq[range(n // 2)]
    
    Y = np.fft.fft(data) / n
    Y = Y[range(n // 2)]
    return frq, abs(Y)
    
    # Plot the spectrum
    #plt.plot(frq, abs(Y))
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Amplitude')
    #plt.title('Spectrum Analysis')
    #plt.show()

if __name__ == '__main__':
    HPFreq, HPSpec = spectrum_analysis('HPHallo.m4a')
    fig1=plt.subplot(211)
    plt.plot(HPFreq,HPSpec)
    plt.title("HP Voice Spectrum")
    XPFreq, XPSpec = spectrum_analysis('XPHallo.m4a')
    fig2=plt.subplot(212)
    plt.plot(XPFreq,XPSpec)
    plt.title("XP Voice Spectrum")
    plt.show()