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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, pool, serial_port=None, depth_img_only=False, verbose=False):
        super().__init__()

        self.pool = pool

        if serial_port is None:
            if platform == "darwin":  # macOS
                ports = list(serial.tools.list_ports.comports())
                if ports:
                    # Try to auto-select a port with likely USB/Serial in description
                    for port in ports:
                        # print(port.device)
                        if "USB" in port.description or "Serial" in port.description:
                            self.serial_port = port.device
                            break
                else:
                    self.serial_port = ports[0].device
            elif platform == "win32":  # Windows
                ports = list(serial.tools.list_ports.comports())
                if ports:
                    # Try to auto-select a port that typically starts with "COM"
                    for port in ports:
                        if "COM" in port.device:
                            self.serial_port = port.device
                            break
                else:
                    self.serial_port = ports[0].device
                
            else:
                self.serial_port = "/dev/ttyACM0"
            print("No serial port provided. Using auto-selected port:", self.serial_port)
        else:
            self.serial_port = serial_port
            print("Using provided serial port:", self.serial_port)

        if self.pool:
            self.num_zones = 1
        else:
            self.num_zones = 9

        self.depth_img_only = depth_img_only
        self.verbose = verbose

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid_layout = QtWidgets.QGridLayout(self.central_widget)

        self.plot_widgets = []
        self.lines = []

        grid_size = int(np.ceil(np.sqrt(self.num_zones)))
        pen = pg.mkPen(color=(255, 255, 255))

        for i in range(self.num_zones):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("k")
            # Set title style to white color for better visibility on black background
            plot_widget.setTitle(title=f"Zone {i+1} (idx={i})", color="w", size="12pt")
            self.plot_widgets.append(plot_widget)
            self.lines.append(plot_widget.plot([], [], pen=pen))
            row = i // grid_size
            col = i % grid_size
            self.grid_layout.addWidget(plot_widget, row, col)

        if not self.pool:
            # Add the image plot to the right of the 4x4 grid
            self.image_plot_widget = pg.PlotWidget()
            self.image_plot_widget.setBackground("k")
            self.image_item = pg.ImageItem()
            self.image_plot_widget.addItem(self.image_item)
            self.grid_layout.addWidget(
                self.image_plot_widget,  # widget to add
                0,  # row (0-indexed)
                grid_size,  # column (0-indexed)
                2,  # row span
                2,  # column span
            )

        if not self.pool:
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

        self.tmf_reader = TMFReader(self.serial_port)

        self.started = False
        self.second_cycle = False
        self.concat_frame_data = {}

    def update_plot(self):
        hists, dists = self.tmf_reader.get_measurement(reset_buffer=False)

        # the first zone is the reference histogram, so remove it
        hists = np.array(hists[0][1:])

        if self.pool:
            hists = np.sum(hists, axis=0)[None, :]

        dists_array = dists[0]["depths_1"]
                    
        print(f"FPS: {self.frame_idx / (time.time() - self.start_time)}")
        self.frame_idx += 1

        if not self.depth_img_only:
            # update line plot data
            for zone in range(self.num_zones):
                self.lines[zone].setData(hists[zone])

        if not self.pool:
            # update image plot data
            self.image_item.setImage(
                np.array(dists_array).reshape(
                int(np.sqrt(self.num_zones)), 
                int(np.sqrt(self.num_zones))
                )
            )


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
        default="/dev/ttyACM0",
        help="Serial port of MCU with VL sensor",
    )
    parser.add_argument(
        "--depth_img_only",
        action="store_true",
        help="Only show the depth image plot, not the individual zone plots (avoids dropped frames)",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Pool data from each zone into a single plot"
    )
    args = parser.parse_args()

    # Set up the Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    app = QtWidgets.QApplication([])
    window = MainWindow(
        serial_port=args.port, pool=args.pool, depth_img_only=args.depth_img_only, verbose=args.verbose
    )
    window.show()
    
    # This allows SIGINT to be processed
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)  # Small timeout to allow signals to be processed
    
    sys.exit(app.exec())
