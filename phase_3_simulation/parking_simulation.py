import time
import random
import numpy as np

ROWS = 5
COLS = 10
TOTAL_SLOTS = ROWS * COLS
TIME_STEPS = 10


class ParkingLotSimulation:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)

    def display_grid(self, timestep):
        print("\n" + "=" * 50)
        print(f"Time Step: {timestep}")
        print("=" * 50)

        for row in self.grid:
            line = ""
            for slot in row:
                if slot == 1:
                    line += "[X] "
                else:
                    line += "[ ] "
            print(line)

        print("\n[X] = Occupied | [ ] = Empty")

    def add_random_cars(self):
        empty_slots = list(zip(*np.where(self.grid == 0)))
        random.shuffle(empty_slots)

        cars_to_add = random.randint(1, 5)

        for position in empty_slots[:cars_to_add]:
            self.grid[position] = 1

    def run(self):
        print("Smart Parking Simulation Started")
        print(f"Parking Lot Size: {ROWS} x {COLS}")
        print(f"Total Slots: {TOTAL_SLOTS}")

        for timestep in range(1, TIME_STEPS + 1):
            self.add_random_cars()
            self.display_grid(timestep)
            time.sleep(0.5)


if __name__ == "__main__":
    simulation = ParkingLotSimulation()
    simulation.run()