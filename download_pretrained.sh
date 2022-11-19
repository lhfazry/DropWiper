#!/opt/conda/bin/python
from pathlib import Path
import gdown
import zipfile
import os

Path("pretrained").mkdir(exist_ok=True)

if not os.path.exists('pretrained/RainDrop_ema_0.9999_1000000.pt'):
    gdown.download(id="1wWMePiseUNCZAD-GuyylwrAmyUtvnPUb", output="models/RainDrop_ema_0.9999_1000000.pt", quiet=False)

if not os.path.exists('pretrained/RainDrop_ema_0.9999_310000.pt'):
    gdown.download(id="1Rnj2Lia4utHGHYRmI8qO3GNXPV1wjlWh", output="models/RainDrop_ema_0.9999_310000.pt", quiet=False)