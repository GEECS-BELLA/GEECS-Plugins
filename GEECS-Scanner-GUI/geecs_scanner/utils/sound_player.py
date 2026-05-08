"""Sound playback utilities for GEECS Scanner.

Includes per-shot beeps, multi-scanner completion jingles, and action
success/failure jingles.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import random
import threading
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


class SoundPlayer(SimpleSoundPlayer):
    """Threaded sound player that queues beep and toot requests."""

    def __init__(
        self,
        beep_frequency=700,
        beep_duration=0.1,
        toot_frequency=1200,
        toot_duration=0.75,
        sample_rate=44100,
        randomized_beeps: bool = False,
    ):
        """Initialize the SoundPlayer and start the background queue thread.

        Parameters
        ----------
        beep_frequency : int, optional
            Frequency of the beep sound in Hz. Default is 700.
        beep_duration : float, optional
            Duration of the beep sound in seconds. Default is 0.1.
        toot_frequency : int, optional
            Frequency of the toot sound in Hz. Default is 1200.
        toot_duration : float, optional
            Duration of the toot sound in seconds. Default is 0.75.
        sample_rate : int, optional
            Sample rate for sound generation (used for macOS). Default is 44100.
        randomized_beeps : bool, optional
            Vary pitch between shots. Default is False.
        """
        super().__init__(sample_rate=sample_rate)

        self.beep_frequency = beep_frequency
        self.beep_duration = beep_duration
        self.toot_frequency = toot_frequency
        self.toot_duration = toot_duration

        self.sound_queue = queue.Queue()

        self.sound_thread = threading.Thread(target=self._process_queue)
        self.sound_thread.daemon = True
        self.running = False

        self.random_beeps = randomized_beeps

    def start_queue(self):
        """Start the background sound processing thread."""
        self.running = True
        self.sound_thread.start()

    def play_beep(self):
        """Add a beep sound request to the queue."""
        self.sound_queue.put("beep")

    def play_toot(self):
        """Add a toot sound request to the queue."""
        self.sound_queue.put("toot")

    def stop(self):
        """Stop the sound player by sending a termination signal."""
        self.running = False
        self.sound_queue.put(None)

    def _process_queue(self):
        """Process the sound queue and play sounds until stopped."""
        while self.running:
            try:
                sound_type = self.sound_queue.get()

                if sound_type is None:
                    break

                if sound_type == "beep":
                    if self.random_beeps:
                        self.play_sound(
                            round(
                                self.beep_frequency * (0.7 * (random.random() + 0.5))
                            ),
                            self.beep_duration,
                        )
                    else:
                        self.play_sound(self.beep_frequency, self.beep_duration)
                elif sound_type == "toot":
                    self.play_sound(self.toot_frequency, self.toot_duration)
                self.sound_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing sound: {e}")


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
