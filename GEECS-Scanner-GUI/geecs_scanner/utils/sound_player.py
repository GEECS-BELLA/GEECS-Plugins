from __future__ import annotations
from typing import Optional

import platform
import os
import queue
import threading
import numpy as np
import logging
import time
import random

# For Windows-specific imports
if platform.system() == "Windows":
    import winsound
# For macOS-specific imports
elif platform.system() == "Darwin":
    import simpleaudio as sa


class SimpleSoundPlayer:
    """ Un-threaded sound player.  Only capable of receiving a playing single notes """
    def __init__(self, sample_rate=44100):
        """
        Args
            sample_rate (int, optional): Sample rate for sound generation (used for macOS). Default is 44100.
        """
        self.sample_rate = sample_rate

    def play_sound(self, frequency, duration):
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
            audio_data = self._generate_sound(frequency, duration)
            play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)  # 1 channel, 2 bytes per sample
            play_obj.wait_done()
        # Optionally add Linux support or other platforms if needed
        else:
            os.system('printf "\a"')  # Default to terminal bell for unsupported platforms

    def _generate_sound(self, frequency, duration):

        """
        Generate a sound for macOS given a frequency and duration.

        Args:
            frequency (int): Frequency of the sound in Hz.
            duration (float): Duration of the sound in seconds.

        Returns:
            numpy.ndarray: Array of sound data formatted for playback.
        """

        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        return (tone * 32767).astype(np.int16)  # Convert to 16-bit PCM format


class SoundPlayer(SimpleSoundPlayer):
    """
    A class to handle playing sounds (beep and toot) in a background thread.
    """

    def __init__(self, beep_frequency=700, beep_duration=0.1, toot_frequency=1200, toot_duration=0.75,
                 sample_rate=44100, options: Optional[dict] = None):
        """
        Initialize the SoundPlayer with default or user-defined frequency, duration.  Then begins thread and Queue

        Args:
            beep_frequency (int, optional): Frequency of the beep sound in Hz. Default is 500.
            beep_duration (float, optional): Duration of the beep sound in seconds. Default is 0.1.
            toot_frequency (int, optional): Frequency of the toot sound in Hz. Default is 1500.
            toot_duration (float, optional): Duration of the toot sound in seconds. Default is 0.75.
            sample_rate (int, optional): Sample rate for sound generation (used for macOS). Default is 44100.
        """
        super().__init__(sample_rate=sample_rate)

        # Sets the frequency and duration of the scan beeps and toots
        self.beep_frequency = beep_frequency
        self.beep_duration = beep_duration
        self.toot_frequency = toot_frequency
        self.toot_duration = toot_duration

        # Create a queue to hold sound requests
        self.sound_queue = queue.Queue()

        # Create and start the background thread
        self.sound_thread = threading.Thread(target=self._process_queue)
        self.sound_thread.daemon = True  # Mark thread as a daemon so it exits when the main program exits
        self.running = False  # Flag to control thread running

        self.random_beeps = False if options is None else options.get('randomized_beeps', False)

    def start_queue(self):
        self.running = True  # Flag to control thread running
        self.sound_thread.start()

    def play_beep(self):
        """Add a beep sound request to the queue."""
        self.sound_queue.put('beep')

    def play_toot(self):
        """Add a toot sound request to the queue."""
        self.sound_queue.put('toot')

    def stop(self):
        """Stop the sound player by sending a termination signal."""
        self.running = False
        self.sound_queue.put(None)  # Add a termination signal to the queue

    def _process_queue(self):

        """
        Continuously process the sound queue and play the appropriate sound
        based on the request.
        """

        while self.running:
            try:
                # Wait for the next sound request (this blocks until a request is added)
                sound_type = self.sound_queue.get()

                # Exit the loop if the termination signal is received
                if sound_type is None:
                    break

                # Play the requested sound
                if sound_type == 'beep':
                    if self.random_beeps:
                        self.play_sound(round(self.beep_frequency*(0.7*(random.random()+0.5))), self.beep_duration)
                    else:
                        self.play_sound(self.beep_frequency, self.beep_duration)
                elif sound_type == 'toot':
                    self.play_sound(self.toot_frequency, self.toot_duration)
                # Mark the task as done
                self.sound_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing sound: {e}")


def multiscan_finish_jingle():
    """Play a jingle of 4 notes, used at the end of a multiscan script"""
    notes = [
        (784, 0.25, 0),
        (1047, 0.10, 0.15),
        (1175, 0.25, 0),
        (1568, 0.5, 0)
    ]
    player = SimpleSoundPlayer()
    for freq, duration, wait in notes:
        player.play_sound(freq, duration)
        time.sleep(wait)


if __name__ == '__main__':
    sound_player = SoundPlayer(beep_frequency=800, toot_frequency=2000)
    sound_player.play_toot()
    sound_player._process_queue()
    time.sleep(1)
    sound_player.stop()
