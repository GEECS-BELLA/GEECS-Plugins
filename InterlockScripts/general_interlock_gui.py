"""Launch the experiment-independent GEECS interlock builder GUI.

Reads SERVER_IP / SERVER_PORT from a .env file in this directory.
"""

from dotenv import load_dotenv

from geecs_python_api.controls.interlocks.interlock_gui import (
    launch_general_interlock_gui,
)


if __name__ == "__main__":
    load_dotenv()
    launch_general_interlock_gui()
