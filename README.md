# motion-marmot

The marmot serves as a motion detector which can target out all possible motions.

<img src="https://media.giphy.com/media/ROcSJHrOhhBkc/source.gif" title="Motion Marmot" width=20% align="right">
When every time it detects motion, it will be like...

## Overview

The motion filter can possibly filter out all static frames in order to reduce bandwidth costs and FP rate in the surveillance system with Computer Vision. Mis-detection may frequently happen and consequently increase the FP rate. Especially in the surveillance system, the constant video streaming may cost a lot of FP detection due to the same case and the file transportation may have huge amount of costs as well. The solution with MotionMarmot may effectively achieve the cost reduction.

## Prerequisite

### Package Installment

```bash
python3 setup.py install [--user] # --user is for either installing in user local side or global.
```

## Usage

### Using Visdom

```bash
python3 motion-marmot/visualization/visdom_playground.py video_clip.mp4
```

## Acknowledgement

-   [NumPy](https://numpy.org/)
-   [OpenCV Python](https://github.com/opencv/opencv-python)
-   [Scikit Learn](https://scikit-learn.org/)
-   [Pandas](https://pandas.pydata.org/)
-   [Typer](https://github.com/tiangolo/typer)
-   [Visdom](https://github.com/fossasia/visdom)
