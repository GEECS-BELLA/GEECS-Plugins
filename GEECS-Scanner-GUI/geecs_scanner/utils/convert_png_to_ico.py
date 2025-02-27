"""
Small script to convert a png image to an ico file, which is needed for setting the Windows icon for the application
"""

from PIL import Image
from pathlib import Path

png_file = Path("../app/gui/application_icon.png")
img = Image.open(png_file)

ico_file = png_file.with_suffix(".ico")
img.save(ico_file, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
