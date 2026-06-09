import time
import random
import numpy as np

ROWS = 5
COLS = 10
TOTAL_SLOTS = ROWS * COLS
TIME_STEPS = 15


class ParkingLotSimulation:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)

    def get_occupancy_rate(self):
        occupied_slots = np.sum(self.grid)
        return occupied_slots / TOTAL_SLOTS

    def display_grid(self, timestep, cars_arrived, cars_left):
        print("\n" + "=" * 60)
        print(f"Time Step: {timestep}")
        print(f"Cars Arrived: {cars_arrived}")
        print(f"Cars Left: {cars_left}")
        print(f"Occupancy Rate: {self.get_occupancy_rate() * 100:.2f}%")
        print("=" * 60)

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
        cars_parked = 0

        for position in empty_slots[:cars_to_add]:
            self.grid[position] = 1
            cars_parked += 1

        return cars_parked

    def remove_random_cars(self):
        occupied_slots = list(zip(*np.where(self.grid == 1)))
        random.shuffle(occupied_slots)

        if not occupied_slots:
            return 0

        cars_to_remove = random.randint(0, 3)
        cars_left = 0

        for position in occupied_slots[:cars_to_remove]:
            self.grid[position] = 0
            cars_left += 1

        return cars_left

    def run(self):
        print("Smart Parking Simulation Started")
        print(f"Parking Lot Size: {ROWS} x {COLS}")
        print(f"Total Slots: {TOTAL_SLOTS}")

        for timestep in range(1, TIME_STEPS + 1):
            cars_left = self.remove_random_cars()
            cars_arrived = self.add_random_cars()

            self.display_grid(timestep, cars_arrived, cars_left)
            time.sleep(0.5)


if __name__ == "__main__":
    simulation = ParkingLotSimulation()
    simulation.run()