# Single Photon Imaging Hands-On Demo
Materials for hands-on portion of ICCP 2025 Summer School on Single Photon Imaging

# Getting Started
## Hardware

Assemble the hardware setup as shown below using the provided parts. The USB and I2C cables are reversible.

Parts:

[AMS TMF8820 Sensor](https://www.sparkfun.com/sparkfun-qwiic-mini-dtof-imager-tmf8820.html)

[SparkFun Qwiic Pro Micro](https://www.sparkfun.com/sparkfun-qwiic-pro-micro-usb-c-atmega32u4.html)

[SparkFun Qwiic Cable](https://www.sparkfun.com/qwiic-cable-100mm.html)

USB C to C cable

![Hardware setup](media/hardware_setup.png)

## Software
The microcontroller has already been flashed with [custom firmware](https://github.com/uwgraphics/ProximityPlanarRecovery/tree/main/arduino) so that it forwards on measurements from the TMF8820 to the connected computer. All you need to do is plug it in.

The `live_vis.py` and `capture.py` scripts can be used to visualize and to record data from the sensor, respectively. In order to run these scripts, you need to install the appropriate python packages. 

### Conda Installation (Preferred)
First, [install Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) if you haven't.

Next, set up the environment
```bash
conda create -n SPAD-ICCP-Summer-School python=3.12.10
conda activate SPAD-ICCP-Summer-School
conda install numpy pyqtgraph pyserial
pip install pyqt6
```

# Activities
See separate files for instructions on [Activity 1](activity1.md) and [Activity 2](activity2.md).