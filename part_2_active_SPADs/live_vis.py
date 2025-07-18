"""
Live update a grid of subplots and a single image plot, side-by-side
"""

import argparse
import sys
import time
import signal

from sys import platform
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
import serial
from PyQt6 import QtCore, QtWidgets

from tmf_reader import TMFReader
from activity3 import estimate_distance as estimate_distance_fn


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, serial_port=None, verbose=False, estimate_distance=False):
        super().__init__()

        self.estimate_distance = estimate_distance

        self.serial_port = serial_port
        self.auto_select_serial_port()

        self.num_zones = 9

        self.verbose = verbose

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QtWidgets.QGridLayout(self.central_widget)

        self.plot_widgets = []
        self.lines = []
        self.pooled_plot_widget = None
        self.pooled_lines = None

        grid_size = int(np.ceil(np.sqrt(self.num_zones)))
        pen = pg.mkPen(color=(255, 255, 255))

        for i in range(self.num_zones):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("k")
            # Set title style to white color for better visibility on black background
            plot_widget.setTitle(title=f"Zone {i+1}", color="w", size="12pt")
            # Add x-axis label with enhanced visibility
            plot_widget.setLabel("bottom", "Bin Index", color="w", size="12pt")
            plot_widget.showAxis("bottom", True)
            plot_widget.getAxis("bottom").setTextPen("w")
            self.plot_widgets.append(plot_widget)
            self.lines.append(plot_widget.plot([], [], pen=pen))
            row = i // grid_size
            col = i % grid_size
            self.grid_layout.addWidget(plot_widget, row, col)

        # Add the image plot to the right of the 4x4 grid
        self.pooled_plot_widget = pg.PlotWidget()
        self.pooled_plot_widget.setBackground("k")
        self.pooled_lines = self.pooled_plot_widget.plot([], [], pen=pen)
        self.pooled_plot_widget.setLabel("bottom", "Bin Index", color="w", size="12pt")
        self.pooled_plot_widget.showAxis("bottom", True)
        self.pooled_plot_widget.getAxis("bottom").setTextPen("w")
        self.pooled_plot_widget.setTitle(title=f"Pooled (sum of all zones per-bin)", color="w", size="12pt")
        self.grid_layout.addWidget(
            self.pooled_plot_widget,  # widget to add
            0,  # row (0-indexed)
            grid_size,  # column (0-indexed)
            2,  # row span
            2,  # column span
        )

        # Set column stretch factors
        for col in range(grid_size):
            self.grid_layout.setColumnStretch(col, 1)
        self.grid_layout.setColumnStretch(grid_size, 3)  # Make the image column wider

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.start_time = time.time()
        self.frame_idx = 0

        try:
            self.tmf_reader = TMFReader(self.serial_port)
        except Exception as e:
            print(f"Error initializing - is the port ({self.serial_port}) correct?")
            print("Tip: pass in the port manually with the -p flag. e.g.: python live_vis.py -p /dev/ttyACM0")
            exit()

        self.started = False
        self.second_cycle = False
        self.concat_frame_data = {}

    def update_plot(self):
        hists, dists = self.tmf_reader.get_measurement(reset_buffer=False)

        # the first zone is the reference histogram, so remove it
        hists = np.array(hists[0][1:])
        pooled_hists = np.sum(hists, axis=0)

        dists_array = dists[0]["depths_1"]

        # print(f"FPS: {self.frame_idx / (time.time() - self.start_time)}")
        self.frame_idx += 1

        if self.estimate_distance:
            self.pooled_lines.setData(hists[4])
            self.pooled_plot_widget.setTitle(
                title=f"Center Zone Histogram | Onboard Distance: {dists_array[4]}mm | Your Algorithm: {estimate_distance_fn(hists[4])*1000:.0f}mm"
            )
            self.pooled_plot_widget.setYRange(0, hists[4].max())
        else:
            self.pooled_lines.setData(pooled_hists)
            self.pooled_plot_widget.setYRange(0, pooled_hists.max())

        # update line plot data
        for zone in range(self.num_zones):
            self.lines[zone].setData(hists[zone])
            self.plot_widgets[zone].setYRange(0, hists[zone].max())

    def auto_select_serial_port(self):
        if self.serial_port is None:
            if platform == "darwin":  # macOS
                ports = list(serial.tools.list_ports.comports())
                if ports:
                    for port in ports:
                        if "SparkFun" in port.description:
                            self.serial_port = port.device
                            print("Using auto-selected serial port:", self.serial_port)
                            return
                    self.serial_port = ports[0].device
                    print("Warning: No SparkFun port found. Check connection. Using first port:", self.serial_port)
            elif platform == "win32":  # Windows
                ports = list(serial.tools.list_ports.comports())
                if ports:
                    for port in ports:
                        if "USB Serial Device" in port.description or "SparkFun" in port.description:
                            self.serial_port = port.device
                            print("Using auto-selected serial port:", self.serial_port)
                            return
                    self.serial_port = ports[0].device
                    print("Warning: No SparkFun port found. Check connection. Using first port:", self.serial_port)
            else: # linux
                ports = list(serial.tools.list_ports.comports())
                if ports:
                    for port in ports:
                        if "SparkFun" in port.description:
                            self.serial_port = port.device
                            print("Using auto-selected serial port:", self.serial_port)
                            return
                    self.serial_port = ports[0].device
                    print("Warning: No SparkFun port found. Check connection. Using first port:", self.serial_port)
        else:
            print("Using provided serial port:", self.serial_port)



def signal_handler(sig, frame):
    print("\nCaught Ctrl+C, exiting...")
    QtWidgets.QApplication.quit()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output (for debugging)"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=str,
        help="Serial port of MCU with VL sensor",
    )
    parser.add_argument(
        "--estimate_distance",
        action="store_true",
        help="Estimate distance from the histogram using your algorithm (implemented in activity3.py)",
    )
    args = parser.parse_args()

    # Set up the Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    app = QtWidgets.QApplication([])
    window = MainWindow(
        serial_port=args.port,
        verbose=args.verbose,
        estimate_distance=args.estimate_distance,
    )
    window.show()

    # This allows SIGINT to be processed
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)  # Small timeout to allow signals to be processed

    sys.exit(app.exec())
