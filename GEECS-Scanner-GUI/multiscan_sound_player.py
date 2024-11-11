"""
Module with a lightweight audio player for sounds specific to the GEECS Scanner GUI and its associated gui windows

-Chris
"""

import platform
import os
import time
import numpy as np

# For Windows-specific imports
if platform.system() == "Windows":
    import winsound
# For macOS-specific imports
elif platform.system() == "Darwin":
    import simpleaudio as sa

SAMPLE_RATE = 44100


def play_finish_jingle():
    """Play a jingle of 4 notes, used at the end of a multiscan script"""
    notes = [
        (784, 0.25, 0),
        (1047, 0.10, 0.15),
        (1175, 0.25, 0),
        (1568, 0.5, 0)
    ]
    for freq, duration, wait in notes:
        play_sound(freq, duration)
        time.sleep(wait)


def play_sound(frequency, duration):
    """
    Play a sound based on the platform (Windows or macOS).

    Args:
        frequency (int): Frequency of the sound in Hz.
        duration (float): Duration of the sound in seconds.
    """
    # Windows: Use winsound.Beep
    if platform.system() == "Windows":
        winsound.Beep(frequency, int(duration * 1000))  # Duration is in milliseconds
    # macOS: Use simpleaudio to play the generated sound
    elif platform.system() == "Darwin":
        audio_data = generate_sound(frequency, duration)
        play_obj = sa.play_buffer(audio_data, 1, 2, SAMPLE_RATE)  # 1 channel, 2 bytes per sample
        play_obj.wait_done()
    # Optionally add Linux support or other platforms if needed
    else:
        os.system('printf "\a"')  # Default to terminal bell for unsupported platforms


def generate_sound(frequency, duration):

    """
    Generate a sound for macOS given a frequency and duration.

    Args:
        frequency (int): Frequency of the sound in Hz.
        duration (float): Duration of the sound in seconds.

    Returns:
        numpy.ndarray: Array of sound data formatted for playback.
    """

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return (tone * 32767).astype(np.int16)  # Convert to 16-bit PCM format
