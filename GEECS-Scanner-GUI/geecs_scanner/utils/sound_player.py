"""Sound playback utilities for GEECS Scanner.

Multi-scanner completion jingles and action success/failure jingles.
(The threaded per-shot ``SoundPlayer`` died with the legacy engine — G1;
per-shot beeps live in GEECS-Console now.)
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np

# For Windows-specific imports
if platform.system() == "Windows":
    import winsound
# For macOS-specific imports
elif platform.system() == "Darwin":
    import simpleaudio as sa


class SimpleSoundPlayer:
    """Un-threaded sound player capable of playing single notes."""

    def __init__(self, sample_rate=44100):
        """Initialize the player with the given sample rate.

        Parameters
        ----------
        sample_rate : int, optional
            Sample rate for sound generation (used for macOS). Default is 44100.
        """
        self.sample_rate = sample_rate

    def play_sound(self, frequency, duration):
        """Play a sound based on the platform (Windows or macOS).

        Parameters
        ----------
        frequency : int
            Frequency of the sound in Hz.
        duration : float
            Duration of the sound in seconds.
        """
        if platform.system() == "Windows":
            winsound.Beep(frequency, int(duration * 1000))
        elif platform.system() == "Darwin":
            audio_data = self._generate_sound(frequency, duration)
            play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)
            play_obj.wait_done()
        else:
            os.system('printf "\a"')

    def _generate_sound(self, frequency, duration):
        """Generate a sound for macOS given a frequency and duration.

        Parameters
        ----------
        frequency : int
            Frequency of the sound in Hz.
        duration : float
            Duration of the sound in seconds.

        Returns
        -------
        numpy.ndarray
            Array of sound data formatted for playback.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        return (tone * 32767).astype(np.int16)


def play_jingle(notes: list[tuple[int, float, float]]):
    """Play a given sequence of notes.

    Parameters
    ----------
    notes : list of tuple[int, float, float]
        Each tuple is ``(frequency_hz, duration_s, wait_s)``.
    """
    player = SimpleSoundPlayer()
    for freq, duration, wait in notes:
        player.play_sound(freq, duration)
        time.sleep(wait)


def multiscan_finish_jingle():
    """Play a 4-note jingle used at the end of a multiscan script."""
    notes = [(784, 0.25, 0), (1047, 0.10, 0.15), (1175, 0.25, 0), (1568, 0.5, 0)]
    play_jingle(notes)


def action_finish_jingle():
    """Play a jingle indicating a successful action completion."""
    notes = [
        (900, 0.20, 0),
        (1300, 0.50, 0),
    ]
    play_jingle(notes)


def action_failed_jingle():
    """Play a jingle indicating an action encountered an error."""
    notes = [
        (1300, 0.20, 0),
        (900, 0.50, 0),
    ]
    play_jingle(notes)


if __name__ == "__main__":
    action_finish_jingle()
