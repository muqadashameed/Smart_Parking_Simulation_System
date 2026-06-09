import time
import random
import numpy as np

ROWS = 5
COLS = 10
TOTAL_SLOTS = ROWS * COLS
TIME_STEPS = 20


class ParkingLotSimulation:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)

        self.total_arrived = 0
        self.total_parked = 0
        self.total_left = 0
        self.total_waiting = 0
        self.total_rejected = 0

        self.wait_times = []
        self.occupancy_history = []
        self.peak_congestion = 0

    def get_occupancy_rate(self):
        occupied_slots = np.sum(self.grid)
        return occupied_slots / TOTAL_SLOTS

    def display_grid(self, timestep, cars_arrived, cars_parked, cars_left, cars_waiting, cars_rejected):
        occupancy_rate = self.get_occupancy_rate()

        print("\n" + "=" * 70)
        print(f"Time Step: {timestep}")
        print(f"Cars Arrived: {cars_arrived}")
        print(f"Cars Parked: {cars_parked}")
        print(f"Cars Left: {cars_left}")
        print(f"Cars Waiting: {cars_waiting}")
        print(f"Cars Rejected: {cars_rejected}")
        print(f"Occupancy Rate: {occupancy_rate * 100:.2f}%")
        print("=" * 70)

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

        cars_arrived = random.randint(2, 8)
        cars_parked = 0
        cars_waiting = 0
        cars_rejected = 0

        for _ in range(cars_arrived):
            if empty_slots:
                position = empty_slots.pop()
                self.grid[position] = 1
                cars_parked += 1

                wait_time = random.randint(1, 4)
                self.wait_times.append(wait_time)
            else:
                cars_waiting += 1
                cars_rejected += 1

                wait_time = random.randint(5, 12)
                self.wait_times.append(wait_time)

        self.total_arrived += cars_arrived
        self.total_parked += cars_parked
        self.total_waiting += cars_waiting
        self.total_rejected += cars_rejected

        return cars_arrived, cars_parked, cars_waiting, cars_rejected

    def remove_random_cars(self):
        occupied_slots = list(zip(*np.where(self.grid == 1)))
        random.shuffle(occupied_slots)

        if not occupied_slots:
            return 0

        cars_to_remove = random.randint(0, 4)
        cars_left = 0

        for position in occupied_slots[:cars_to_remove]:
            self.grid[position] = 0
            cars_left += 1

        self.total_left += cars_left
        return cars_left

    def update_metrics(self):
        occupancy_rate = self.get_occupancy_rate()
        self.occupancy_history.append(occupancy_rate)

        if occupancy_rate > self.peak_congestion:
            self.peak_congestion = occupancy_rate

    def show_final_metrics(self):
        average_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        average_occupancy = sum(self.occupancy_history) / len(self.occupancy_history) if self.occupancy_history else 0

        print("\n" + "=" * 70)
        print("FINAL SIMULATION METRICS")
        print("=" * 70)
        print(f"Total Cars Arrived: {self.total_arrived}")
        print(f"Total Cars Parked: {self.total_parked}")
        print(f"Total Cars Left: {self.total_left}")
        print(f"Total Cars Waiting: {self.total_waiting}")
        print(f"Total Cars Rejected: {self.total_rejected}")
        print(f"Average Wait Time: {average_wait_time:.2f} minutes")
        print(f"Average Occupancy Rate: {average_occupancy * 100:.2f}%")
        print(f"Peak Congestion: {self.peak_congestion * 100:.2f}%")
        print("=" * 70)

    def run(self):
        print("Smart Parking Simulation Started")
        print(f"Parking Lot Size: {ROWS} x {COLS}")
        print(f"Total Slots: {TOTAL_SLOTS}")

        for timestep in range(1, TIME_STEPS + 1):
            cars_left = self.remove_random_cars()
            cars_arrived, cars_parked, cars_waiting, cars_rejected = self.add_random_cars()

            self.update_metrics()

            self.display_grid(
                timestep,
                cars_arrived,
                cars_parked,
                cars_left,
                cars_waiting,
                cars_rejected
            )

            time.sleep(0.5)

        self.show_final_metrics()


if __name__ == "__main__":
    simulation = ParkingLotSimulation()
    simulation.run()