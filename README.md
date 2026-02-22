# Dataset Creation

In this repository you can find 2 datasets. The first one consists of a formula 1 motor synthetic generation, and the other one is for violin glissandi. 

Formula 1 is structured with a consistent RPM (revolutions per minute) motor synthesis per file, meaning that csv files only have 1 value.

Violin glissandi consists of 70 frames per second analysis, in which the fundamental frequency is reported. Both are reported, the frequency and the window starting time for the analysis.

## How to run

Both datasets are present here, but in order you want to regenerate, or use your own recordings, here are the steps on how to run the code.

### MOTOR

Run only this command:

```
python motor_generation.py
```

### GLISSANDI

Run in the following order:

```
python analyze_glissando.py path/glissando.wav
python segment_glissandi.py --out_dir raw_glissandi/ glissandi.wav
python create_parameters.py raw_glissandi/
```

As the names indicate, **analyze_glissando.py** creates a csv with the per frame note detection, **segment_glissandi.py** segments the audio with the given csv results obtained. Finally, **create_parameters.py** performs a quite simple statistical analysis and writes the parameters.json associated with the dataset.