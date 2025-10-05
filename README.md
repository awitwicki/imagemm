# ImageMM python realization

Based on paper [Sukurdeep_2025_AJ_170_233](docs/Sukurdeep_2025_AJ_170_233.pdf)

## Run on windows (only in wsl)

1. Place aligned monochrome .fits files to process in `/input` folder

2. Run command to start image rebuilding
```shell
    uv run --with numpy,tensorflow[and-cuda],astropy,matplotlib,scipy,tqdm,scikit-image main.py
```

3. Results are in `/output` folder
