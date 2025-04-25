import time
import serial

TMF882X_CHANNELS = 10
TMF882X_BINS = 128
TMF882X_SKIP_FIELDS = 3  # skip first 3 items in each row
TMF882X_IDX_FIELD = 2  # second item in each row contains the idx field


class TMFReader:
    def __init__(self, port):
        self.arduino = serial.Serial(port=port, baudrate=1000000, timeout=0.1)
        time.sleep(2)

    def get_measurement(self, frames_to_capture=1, reset_buffer=True):
        
        if reset_buffer:
            # clear arduino buffer
            self.arduino.reset_input_buffer()
            frames_finished = -1  # start at -1 because we always throw out the first frame
        else:
            frames_finished = 0

        if frames_to_capture == 0:
            frames_to_capture = 10e10

        buffer = []

        all_processed_hists = []
        all_processed_dists = []

        while frames_finished < frames_to_capture:
            line = self.arduino.readline().rstrip()
            buffer.append(line)
            try:
                decoded_line = line.decode("utf-8").rstrip().split(",")
                if (
                    len(decoded_line) > TMF882X_IDX_FIELD
                    and decoded_line[TMF882X_IDX_FIELD] == "29"
                ):
                    processed_hists = TMFReader.process_raw_hists(buffer)
                    processed_dists = TMFReader.process_raw_dist(buffer)
                    if processed_hists is not None and processed_dists is not None:
                        if frames_finished > -1:
                            all_processed_hists.append(processed_hists)
                            all_processed_dists.append(processed_dists)
                        frames_finished += 1
                    buffer = []

            except UnicodeDecodeError:
                pass  # if you start in a weird place you get random data that can't be decoded, so just ignore
                buffer = []

        return all_processed_hists, all_processed_dists

    @classmethod
    def process_raw_hists(cls, buffer):
        if len(buffer) != 31:
            # print(f"WARNING: Buffer wrong size ({len(buffer)}) - skipping and returning None")
            return None

        rawSum = [[0 for _ in range(TMF882X_BINS)] for _ in range(TMF882X_CHANNELS)]

        for line in buffer:
            data = line.decode("utf-8")
            data = data.replace("\r", "")
            data = data.replace("\n", "")
            row = data.split(",")

            if len(row) > 0 and len(row[0]) > 0 and row[0][0] == "#":
                if (
                    row[0] == "#Raw" and len(row) == TMF882X_BINS + TMF882X_SKIP_FIELDS
                ):  # ignore lines that start with #obj
                    if '' in row:
                        print("Empty entry recieved over serial - skipping and returning None")
                        return None
                    idx = int(
                        row[TMF882X_IDX_FIELD]
                    )  # idx is the id of the histogram (e.g. 0-9 for 9 hists + calibration hist)
                    if idx >= 0 and idx <= 9:
                        for col in range(TMF882X_BINS):
                            rawSum[idx][col] = int(
                                row[TMF882X_SKIP_FIELDS + col]
                            )  # LSB is only assignement
                    elif idx >= 10 and idx <= 19:
                        idx = idx - 10
                        for col in range(TMF882X_BINS):
                            rawSum[idx][col] = (
                                rawSum[idx][col] + int(row[TMF882X_SKIP_FIELDS + col]) * 256
                            )  # mid
                    elif idx >= 20 and idx <= 29:
                        idx = idx - 20
                        for col in range(TMF882X_BINS):
                            rawSum[idx][col] = (
                                rawSum[idx][col] + int(row[TMF882X_SKIP_FIELDS + col]) * 256 * 256
                            )  # MSB

            else:
                print("Incomplete line read - ignoring")

        return rawSum

    @classmethod
    def process_raw_dist(cls, buffer):
        for line in buffer:
            data = line.decode("utf-8")
            data = data.replace("\r", "")
            data = data.replace("\n", "")
            d = data.split(",")

            if len(d) == 78 and d[0] == "#Obj":
                result = {}
                result["I2C_address"] = int(d[1])
                result["measurement_num"] = int(d[2])
                result["temperature"] = int(d[3])
                result["num_valid_results"] = int(d[4])
                result["tick"] = int(d[5])
                result["depths_1"] = [
                    int(x) for x in [d[6], d[8], d[10], d[12], d[14], d[16], d[18], d[20], d[22]]
                ]
                result["confs_1"] = [
                    int(x) for x in [d[7], d[9], d[11], d[13], d[15], d[17], d[19], d[21], d[23]]
                ]
                # 18 that go in between here are unused, at least in 3x3 mode
                result["depths_2"] = [
                    int(x) for x in [d[42], d[44], d[46], d[48], d[50], d[52], d[54], d[56], d[58]]
                ]
                result["confs_2"] = [
                    int(x) for x in [d[43], d[45], d[47], d[49], d[51], d[53], d[55], d[57], d[59]]
                ]
                # last 18 are unused, at least in 3x3 mode

                return result
        return None
