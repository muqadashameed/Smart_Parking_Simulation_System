import os
import math
import random
import joblib
import numpy as np
import pandas as pd

from vpython import (
    box,
    color,
    vector,
    rate,
    scene,
    cylinder,
    sphere,
    compound,
    arrow,
    mag,
    norm,
    label,
)

# ============================================================
# REALISTIC TOP VIEW 3D SMART PARKING BASEMENT SIMULATION
# Features:
# - light concrete basement theme
# - entry/exit sign boards
# - boom barriers
# - green/red parking sensors
# - safety blocks around pillars
# - proper entry/parking/exit route
# - no roof
# - clean top view
# ============================================================

ROWS = 8
COLS = 24
TOTAL_SLOTS = ROWS * COLS
TIME_STEPS = 18

INTRO_SECONDS = 2

LEFT_COLS = 12
RIGHT_COLS = 12

SLOT_WIDTH = 2.05
SLOT_LENGTH = 3.42
SLOT_GAP_X = 0.28

CENTER_AISLE_WIDTH = 4.55
BLOCK_AISLE_WIDTH = 4.35
BLOCK_SPACING_Z = 12.35

CAR_WIDTH = 1.05
CAR_LENGTH = 2.10
CAR_HEIGHT = 0.52
CAR_Y = 0.55

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_parking_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "outputs", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "outputs", "scaler.pkl")

FEATURES = [
    "Occupancy_Rate",
    "Sensor_Reading_Proximity",
    "Sensor_Reading_Pressure",
    "Sensor_Reading_Ultrasonic",
    "Reserved_Status",
    "Hour",
    "Is_Weekend",
    "Nearby_Traffic_Level",
    "Parking_Lot_Section",
    "Spot_Size",
    "Dynamic_Pricing_Factor",
    "Weather_Temperature",
    "Weather_Precipitation",
]

CAR_COLORS = [
    vector(0.82, 0.08, 0.08),
    vector(0.08, 0.22, 0.82),
    vector(0.08, 0.55, 0.18),
    vector(0.95, 0.72, 0.08),
    vector(0.58, 0.58, 0.60),
    vector(0.16, 0.16, 0.18),
    vector(0.75, 0.34, 0.08),
    vector(0.54, 0.20, 0.72),
    vector(0.95, 0.95, 0.95),
]

ZONE_COLORS = [
    vector(0.28, 0.56, 0.95),
    vector(0.28, 0.75, 0.40),
    vector(0.95, 0.60, 0.24),
    vector(0.70, 0.45, 0.95),
]

SENSOR_GREEN = vector(0.05, 0.85, 0.20)
SENSOR_RED = vector(0.90, 0.05, 0.05)


class RealisticBasementParkingSimulation:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)

        self.car_objects = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.sensor_lights = [[None for _ in range(COLS)] for _ in range(ROWS)]

        self.slot_positions = []
        self.block_aisle_z_values = []

        self.total_arrived = 0
        self.total_parked = 0
        self.total_left = 0
        self.total_waiting = 0
        self.total_rejected = 0

        self.wait_times = []
        self.occupancy_history = []
        self.peak_congestion = 0

        self.scene_center = vector(0, 0, 0)

        self.dataset = self.load_dataset()
        self.model = self.load_model()
        self.scaler = self.load_scaler()

        self.entry_position = None
        self.exit_position = None
        self.front_lane_z = None

        self.entry_barrier = None
        self.exit_barrier = None
        self.entry_barrier_pivot = None
        self.exit_barrier_pivot = None
        self.entry_barrier_open = False
        self.exit_barrier_open = False

        self.build_scene()

    # ============================================================
    # DATA + ML
    # ============================================================

    def load_dataset(self):
        df = pd.read_csv(DATA_PATH)

        if df["Nearby_Traffic_Level"].dtype == "object":
            df["Nearby_Traffic_Level"] = df["Nearby_Traffic_Level"].map(
                {"Low": 0, "Medium": 1, "High": 2}
            )

        if df["Parking_Lot_Section"].dtype == "object":
            df["Parking_Lot_Section"] = df["Parking_Lot_Section"].map(
                {"Zone A": 0, "Zone B": 1, "Zone C": 2, "Zone D": 3}
            )

        if df["Spot_Size"].dtype == "object":
            df["Spot_Size"] = df["Spot_Size"].map(
                {"Compact": 0, "Standard": 1, "Oversized": 2}
            )

        df["Is_Weekend"] = df["Is_Weekend"].astype(int)
        df = df.dropna(subset=FEATURES)

        print("Dataset loaded successfully.")
        print(f"Rows available for simulation: {len(df)}")
        return df

    def load_model(self):
        print("Random Forest model loaded successfully.")
        return joblib.load(MODEL_PATH)

    def load_scaler(self):
        print("Scaler loaded successfully.")
        return joblib.load(SCALER_PATH)

    def get_occupancy_rate(self):
        return np.sum(self.grid) / TOTAL_SLOTS

    def predict_demand(self, hour):
        sample = self.dataset.sample(1).copy()

        sample["Hour"] = hour
        sample["Occupancy_Rate"] = self.get_occupancy_rate()

        x = sample[FEATURES]
        x_scaled = self.scaler.transform(x)

        prediction = self.model.predict(x_scaled)[0]
        return prediction

    def calculate_arrivals(self, hour):
        prediction = self.predict_demand(hour)

        if prediction == 1:
            cars_arrived = random.randint(4, 7)
            demand_status = "High Demand"
        else:
            cars_arrived = random.randint(1, 4)
            demand_status = "Low Demand"

        if 8 <= hour <= 11 or 17 <= hour <= 20:
            cars_arrived += random.randint(1, 2)
            demand_status += " + Peak Hour"

        return cars_arrived, demand_status

    # ============================================================
    # LAYOUT
    # ============================================================

    def compute_slot_positions(self):
        positions = []

        block_count = ROWS // 2
        start_z = -((block_count - 1) * BLOCK_SPACING_Z) / 2

        self.block_aisle_z_values = [
            start_z + i * BLOCK_SPACING_Z for i in range(block_count)
        ]

        for r in range(ROWS):
            block_index = r // 2
            is_top_row = r % 2 == 0
            block_aisle_z = self.block_aisle_z_values[block_index]

            if is_top_row:
                slot_z = block_aisle_z - (BLOCK_AISLE_WIDTH / 2 + SLOT_LENGTH / 2)
            else:
                slot_z = block_aisle_z + (BLOCK_AISLE_WIDTH / 2 + SLOT_LENGTH / 2)

            row_positions = []

            for c in range(COLS):
                if c < LEFT_COLS:
                    x = -(
                        CENTER_AISLE_WIDTH / 2
                        + (LEFT_COLS - c - 0.5) * (SLOT_WIDTH + SLOT_GAP_X)
                    )
                else:
                    right_index = c - LEFT_COLS
                    x = (
                        CENTER_AISLE_WIDTH / 2
                        + (right_index + 0.5) * (SLOT_WIDTH + SLOT_GAP_X)
                    )

                row_positions.append(vector(x, CAR_Y, slot_z))

            positions.append(row_positions)

        return positions

    # ============================================================
    # SCENE
    # ============================================================

    def build_scene(self):
        scene.title = ""
        scene.width = 1450
        scene.height = 780
        scene.background = vector(0.90, 0.94, 0.98)
        scene.center = self.scene_center
        scene.range = 28

        # Top-view style. No auto camera movement.
        scene.userspin = False
        scene.userzoom = True
        scene.userpan = True

        self.slot_positions = self.compute_slot_positions()

        all_x = [pos.x for row in self.slot_positions for pos in row]
        all_z = [pos.z for row in self.slot_positions for pos in row]

        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)

        lot_width = (max_x - min_x) + SLOT_WIDTH + 9
        lot_depth = (max_z - min_z) + SLOT_LENGTH + 10

        self.front_lane_z = max_z + SLOT_LENGTH / 2 + 5.0
        self.entry_position = vector(min_x + 2.0, CAR_Y, self.front_lane_z)
        self.exit_position = vector(max_x - 2.0, CAR_Y, self.front_lane_z)

        # Concrete floor
        box(
            pos=vector(0, -0.15, 0),
            size=vector(lot_width, 0.30, lot_depth),
            color=vector(0.80, 0.80, 0.78),
        )

        # Central driving aisle
        box(
            pos=vector(0, 0.02, 0),
            size=vector(CENTER_AISLE_WIDTH, 0.05, lot_depth - 8),
            color=vector(0.50, 0.50, 0.52),
        )

        # Front lane
        box(
            pos=vector(0, 0.02, self.front_lane_z),
            size=vector(lot_width - 4, 0.05, 3.9),
            color=vector(0.50, 0.50, 0.52),
        )

        # Entry and exit ramps
        box(
            pos=vector(self.entry_position.x, 0.03, self.front_lane_z),
            size=vector(6.8, 0.06, 3.9),
            color=vector(0.56, 0.56, 0.58),
        )

        box(
            pos=vector(self.exit_position.x, 0.03, self.front_lane_z),
            size=vector(6.8, 0.06, 3.9),
            color=vector(0.56, 0.56, 0.58),
        )

        # Lane dividers
        box(
            pos=vector(0, 0.08, self.front_lane_z),
            size=vector(lot_width - 10, 0.04, 0.10),
            color=vector(1.0, 0.78, 0.08),
        )

        box(
            pos=vector(0, 0.08, 0),
            size=vector(0.10, 0.04, lot_depth - 12),
            color=vector(1.0, 0.78, 0.08),
        )

        # Block aisles and zone strips
        for index, aisle_z in enumerate(self.block_aisle_z_values):
            zone_color = ZONE_COLORS[index]

            box(
                pos=vector(0, 0.03, aisle_z),
                size=vector(lot_width - 7, 0.055, BLOCK_AISLE_WIDTH),
                color=vector(0.52, 0.52, 0.54),
            )

            box(
                pos=vector(0, 0.09, aisle_z - BLOCK_AISLE_WIDTH / 2 + 0.22),
                size=vector(lot_width - 10, 0.045, 0.14),
                color=zone_color,
            )

            box(
                pos=vector(0, 0.09, aisle_z + BLOCK_AISLE_WIDTH / 2 - 0.22),
                size=vector(lot_width - 10, 0.045, 0.14),
                color=zone_color,
            )

        # Low boundaries
        wall_color = vector(0.84, 0.84, 0.82)

        box(
            pos=vector(0, 0.65, min_z - SLOT_LENGTH / 2 - 4.0),
            size=vector(lot_width, 1.3, 0.35),
            color=wall_color,
        )

        box(
            pos=vector(min_x - SLOT_WIDTH / 2 - 5.0, 0.65, -1),
            size=vector(0.35, 1.3, lot_depth - 7),
            color=wall_color,
        )

        box(
            pos=vector(max_x + SLOT_WIDTH / 2 + 5.0, 0.65, -1),
            size=vector(0.35, 1.3, lot_depth - 7),
            color=wall_color,
        )

        # ENTRY gate and sign
        box(
            pos=vector(self.entry_position.x, 0.55, self.entry_position.z - 1.75),
            size=vector(3.2, 1.1, 0.20),
            color=vector(0.08, 0.82, 0.25),
            opacity=0.95,
        )

        label(
            pos=vector(self.entry_position.x, 1.35, self.entry_position.z - 1.75),
            text="ENTRY",
            height=13,
            box=True,
            color=color.white,
            background=vector(0.08, 0.55, 0.18),
            opacity=0.90,
        )

        # EXIT gate and sign
        box(
            pos=vector(self.exit_position.x, 0.55, self.exit_position.z + 1.75),
            size=vector(3.2, 1.1, 0.20),
            color=vector(0.90, 0.10, 0.10),
            opacity=0.95,
        )

        label(
            pos=vector(self.exit_position.x, 1.35, self.exit_position.z + 1.75),
            text="EXIT",
            height=13,
            box=True,
            color=color.white,
            background=vector(0.65, 0.08, 0.08),
            opacity=0.90,
        )

        # Route arrows
        arrow(
            pos=vector(self.entry_position.x + 1.4, 0.28, self.front_lane_z),
            axis=vector(5.0, 0, 0),
            shaftwidth=0.22,
            color=vector(0.08, 0.70, 0.20),
        )

        arrow(
            pos=vector(self.exit_position.x - 6.2, 0.28, self.front_lane_z),
            axis=vector(5.0, 0, 0),
            shaftwidth=0.22,
            color=vector(0.85, 0.08, 0.08),
        )

        arrow(
            pos=vector(0, 0.28, self.front_lane_z - 1.2),
            axis=vector(0, 0, -5.0),
            shaftwidth=0.20,
            color=vector(1.0, 0.76, 0.06),
        )

        for index, aisle_z in enumerate(self.block_aisle_z_values):
            zone_color = ZONE_COLORS[index]

            arrow(
                pos=vector(1.0, 0.28, aisle_z),
                axis=vector(4.0, 0, 0),
                shaftwidth=0.18,
                color=zone_color,
            )

            arrow(
                pos=vector(-1.0, 0.28, aisle_z),
                axis=vector(-4.0, 0, 0),
                shaftwidth=0.18,
                color=zone_color,
            )

        # Parking slots and parking sensor lights
        for r in range(ROWS):
            block_index = r // 2
            zone_color = ZONE_COLORS[block_index]

            for c in range(COLS):
                pos = self.slot_positions[r][c]

                box(
                    pos=vector(pos.x, 0.03, pos.z),
                    size=vector(SLOT_WIDTH, 0.045, SLOT_LENGTH),
                    color=zone_color,
                    opacity=0.22,
                )

                border_color = vector(0.22, 0.22, 0.22)

                box(
                    pos=vector(pos.x - SLOT_WIDTH / 2, 0.085, pos.z),
                    size=vector(0.04, 0.04, SLOT_LENGTH),
                    color=border_color,
                )

                box(
                    pos=vector(pos.x + SLOT_WIDTH / 2, 0.085, pos.z),
                    size=vector(0.04, 0.04, SLOT_LENGTH),
                    color=border_color,
                )

                box(
                    pos=vector(pos.x, 0.085, pos.z - SLOT_LENGTH / 2),
                    size=vector(SLOT_WIDTH, 0.04, 0.04),
                    color=border_color,
                )

                box(
                    pos=vector(pos.x, 0.085, pos.z + SLOT_LENGTH / 2),
                    size=vector(SLOT_WIDTH, 0.04, 0.04),
                    color=border_color,
                )

                if r % 2 == 0:
                    sensor_z = pos.z + SLOT_LENGTH / 2 - 0.25
                else:
                    sensor_z = pos.z - SLOT_LENGTH / 2 + 0.25

                sensor = sphere(
                    pos=vector(pos.x, 0.24, sensor_z),
                    radius=0.13,
                    color=SENSOR_GREEN,
                    emissive=True,
                )

                self.sensor_lights[r][c] = sensor

        # Pillars with yellow-black safety bases
        pillar_color = vector(0.72, 0.72, 0.70)

        pillar_x_values = [
            min_x + 2.0,
            min_x + 9.0,
            max_x - 9.0,
            max_x - 2.0,
        ]

        for x in pillar_x_values:
            for z in self.block_aisle_z_values:
                box(
                    pos=vector(x, 0.08, z),
                    size=vector(1.45, 0.12, 1.45),
                    color=vector(0.10, 0.10, 0.10),
                )

                for offset in [-0.45, 0.0, 0.45]:
                    box(
                        pos=vector(x + offset, 0.16, z),
                        size=vector(0.18, 0.08, 1.45),
                        color=vector(1.0, 0.75, 0.02),
                    )

                cylinder(
                    pos=vector(x, 0, z),
                    axis=vector(0, 1.4, 0),
                    radius=0.22,
                    color=pillar_color,
                )

        # Ground light strips
        for x in np.linspace(min_x, max_x, 5):
            for z in np.linspace(min_z, max_z, 4):
                box(
                    pos=vector(x, 0.12, z),
                    size=vector(2.8, 0.035, 0.18),
                    color=vector(1.0, 0.92, 0.55),
                    opacity=0.75,
                )

        # Boom barriers
        self.create_barriers()

        # Screen-fit centered top view.
        # Canvas size is kept smaller than the browser width, so it does not crop.
        # Camera is centered from the visible basement frame, not from the browser.
        view_min_x = -lot_width / 2
        view_max_x = lot_width / 2
        view_min_z = min_z - SLOT_LENGTH / 2 - 3.8
        view_max_z = self.front_lane_z + 3.8

        visual_center_x = (view_min_x + view_max_x) / 2
        visual_center_z = (view_min_z + view_max_z) / 2

        view_width = view_max_x - view_min_x
        view_depth = view_max_z - view_min_z
        aspect_ratio = scene.width / scene.height

        scene.center = vector(visual_center_x, 0, visual_center_z)
        scene.range = max(view_depth / 2, view_width / (2 * aspect_ratio)) * 1.08

        scene.camera.pos = vector(visual_center_x, 74, visual_center_z + 0.001)
        scene.camera.axis = vector(0, -74, 0)
        scene.up = vector(0, 0, -1)

    # ============================================================
    # BARRIERS
    # ============================================================

    def create_barriers(self):
        barrier_y = 0.42

        self.entry_barrier_pivot = vector(
            self.entry_position.x + 4.2,
            barrier_y,
            self.front_lane_z - 1.85,
        )

        self.entry_barrier = box(
            pos=self.entry_barrier_pivot + vector(0, 0, 1.85),
            size=vector(0.22, 0.16, 3.7),
            color=vector(0.95, 0.12, 0.12),
        )

        cylinder(
            pos=vector(self.entry_barrier_pivot.x, 0, self.entry_barrier_pivot.z),
            axis=vector(0, 0.8, 0),
            radius=0.13,
            color=vector(0.15, 0.15, 0.15),
        )

        self.exit_barrier_pivot = vector(
            self.exit_position.x - 4.2,
            barrier_y,
            self.front_lane_z - 1.85,
        )

        self.exit_barrier = box(
            pos=self.exit_barrier_pivot + vector(0, 0, 1.85),
            size=vector(0.22, 0.16, 3.7),
            color=vector(0.95, 0.12, 0.12),
        )

        cylinder(
            pos=vector(self.exit_barrier_pivot.x, 0, self.exit_barrier_pivot.z),
            axis=vector(0, 0.8, 0),
            radius=0.13,
            color=vector(0.15, 0.15, 0.15),
        )

    def animate_barrier(self, barrier, pivot, direction):
        for _ in range(18):
            rate(45)
            barrier.rotate(
                angle=direction * (math.pi / 2) / 18,
                axis=vector(0, 1, 0),
                origin=pivot,
            )

    def open_entry_barrier(self):
        if not self.entry_barrier_open:
            self.animate_barrier(self.entry_barrier, self.entry_barrier_pivot, -1)
            self.entry_barrier_open = True

    def close_entry_barrier(self):
        if self.entry_barrier_open:
            self.animate_barrier(self.entry_barrier, self.entry_barrier_pivot, 1)
            self.entry_barrier_open = False

    def open_exit_barrier(self):
        if not self.exit_barrier_open:
            self.animate_barrier(self.exit_barrier, self.exit_barrier_pivot, 1)
            self.exit_barrier_open = True

    def close_exit_barrier(self):
        if self.exit_barrier_open:
            self.animate_barrier(self.exit_barrier, self.exit_barrier_pivot, -1)
            self.exit_barrier_open = False

    # ============================================================
    # VISUAL HELPERS
    # ============================================================

    def update_sensor_lights(self):
        for r in range(ROWS):
            for c in range(COLS):
                sensor = self.sensor_lights[r][c]

                if sensor is None:
                    continue

                if self.grid[r, c] == 1:
                    sensor.color = SENSOR_RED
                else:
                    sensor.color = SENSOR_GREEN

    # ============================================================
    # CAR MODEL
    # ============================================================

    def create_car(self, start_position):
        car_color = random.choice(CAR_COLORS)

        length_scale = random.uniform(0.92, 1.08)
        width_scale = random.uniform(0.92, 1.06)

        car_length = CAR_LENGTH * length_scale
        car_width = CAR_WIDTH * width_scale

        body = box(
            pos=vector(0, 0, 0),
            size=vector(car_length, CAR_HEIGHT, car_width),
            color=car_color,
        )

        roof = box(
            pos=vector(-0.10, 0.38, 0),
            size=vector(car_length * 0.46, CAR_HEIGHT * 0.65, car_width * 0.70),
            color=car_color * 0.82,
        )

        front_glass = box(
            pos=vector(-car_length * 0.31, 0.48, 0),
            size=vector(0.34, 0.04, car_width * 0.58),
            color=vector(0.45, 0.75, 0.95),
            opacity=0.78,
        )

        back_glass = box(
            pos=vector(car_length * 0.24, 0.48, 0),
            size=vector(0.34, 0.04, car_width * 0.58),
            color=vector(0.45, 0.75, 0.95),
            opacity=0.78,
        )

        front_light_left = box(
            pos=vector(-car_length / 2 - 0.02, 0.05, -car_width * 0.25),
            size=vector(0.05, 0.08, 0.18),
            color=vector(1.0, 0.95, 0.60),
            opacity=0.9,
        )

        front_light_right = box(
            pos=vector(-car_length / 2 - 0.02, 0.05, car_width * 0.25),
            size=vector(0.05, 0.08, 0.18),
            color=vector(1.0, 0.95, 0.60),
            opacity=0.9,
        )

        rear_light_left = box(
            pos=vector(car_length / 2 + 0.02, 0.05, -car_width * 0.25),
            size=vector(0.05, 0.08, 0.18),
            color=vector(0.95, 0.05, 0.05),
            opacity=0.9,
        )

        rear_light_right = box(
            pos=vector(car_length / 2 + 0.02, 0.05, car_width * 0.25),
            size=vector(0.05, 0.08, 0.18),
            color=vector(0.95, 0.05, 0.05),
            opacity=0.9,
        )

        wheels = []

        for x in [-car_length * 0.32, car_length * 0.32]:
            for z in [-car_width * 0.55, car_width * 0.55]:
                wheels.append(
                    cylinder(
                        pos=vector(x, -0.30, z),
                        axis=vector(0, 0, 0.18),
                        radius=0.20,
                        color=color.black,
                    )
                )

        car = compound(
            [
                body,
                roof,
                front_glass,
                back_glass,
                front_light_left,
                front_light_right,
                rear_light_left,
                rear_light_right,
            ]
            + wheels
        )

        car.pos = start_position
        return car

    def set_car_direction(self, car, start, end):
        direction = end - start

        if mag(direction) > 0:
            try:
                car.axis = norm(direction)
                car.up = vector(0, 1, 0)
            except Exception:
                pass

    def animate_car(self, car, start, end, steps=36):
        self.set_car_direction(car, start, end)

        for i in range(steps):
            rate(45)
            t = (i + 1) / steps
            car.pos = start + (end - start) * t

    def animate_route(self, car, route, steps_per_segment=32):
        for i in range(len(route) - 1):
            self.animate_car(car, route[i], route[i + 1], steps=steps_per_segment)

    def pause_scene(self, frames=55):
        for _ in range(frames):
            rate(45)

    def intro_pause(self):
        for _ in range(INTRO_SECONDS * 45):
            rate(45)

    # ============================================================
    # ROUTES
    # ============================================================

    def get_route_to_slot(self, target_position, row_index):
        block_index = row_index // 2
        aisle_z = self.block_aisle_z_values[block_index]

        center_front = vector(0, CAR_Y, self.front_lane_z)
        center_block = vector(0, CAR_Y, aisle_z)
        side_aisle = vector(target_position.x, CAR_Y, aisle_z)

        return [
            self.entry_position,
            center_front,
            center_block,
            side_aisle,
            target_position,
        ]

    def get_exit_route_from_slot(self, start_position, row_index):
        block_index = row_index // 2
        aisle_z = self.block_aisle_z_values[block_index]

        side_aisle = vector(start_position.x, CAR_Y, aisle_z)
        center_block = vector(0, CAR_Y, aisle_z)
        center_front = vector(0, CAR_Y, self.front_lane_z)

        return [
            start_position,
            side_aisle,
            center_block,
            center_front,
            self.exit_position,
        ]

    # ============================================================
    # SIMULATION
    # ============================================================

    def add_cars(self, hour):
        empty_slots = list(zip(*np.where(self.grid == 0)))
        random.shuffle(empty_slots)

        cars_arrived, demand_status = self.calculate_arrivals(hour)

        cars_parked = 0
        cars_waiting = 0
        cars_rejected = 0

        for _ in range(cars_arrived):
            if empty_slots:
                r, c = empty_slots.pop()
                target_position = self.slot_positions[r][c]

                car = self.create_car(self.entry_position)
                route = self.get_route_to_slot(target_position, r)

                self.open_entry_barrier()
                self.animate_car(car, route[0], route[1], steps=34)
                self.close_entry_barrier()

                self.animate_route(car, route[1:], steps_per_segment=30)

                self.grid[r, c] = 1
                self.car_objects[r][c] = car
                self.update_sensor_lights()

                cars_parked += 1
                self.wait_times.append(random.randint(1, 4))

            else:
                cars_waiting += 1
                cars_rejected += 1
                self.wait_times.append(random.randint(5, 12))

        self.total_arrived += cars_arrived
        self.total_parked += cars_parked
        self.total_waiting += cars_waiting
        self.total_rejected += cars_rejected

        return cars_arrived, cars_parked, cars_waiting, cars_rejected, demand_status

    def remove_cars(self, hour):
        occupied_slots = list(zip(*np.where(self.grid == 1)))
        random.shuffle(occupied_slots)

        if not occupied_slots:
            return 0

        occupancy = self.get_occupancy_rate()

        if occupancy >= 0.85:
            cars_to_remove = random.randint(3, 6)
        elif occupancy >= 0.60:
            cars_to_remove = random.randint(2, 4)
        elif 13 <= hour <= 18:
            cars_to_remove = random.randint(1, 3)
        else:
            cars_to_remove = random.randint(0, 2)

        cars_to_remove = min(cars_to_remove, len(occupied_slots))
        cars_left = 0

        for r, c in occupied_slots[:cars_to_remove]:
            car = self.car_objects[r][c]

            if car is not None:
                start_position = self.slot_positions[r][c]
                route = self.get_exit_route_from_slot(start_position, r)

                self.animate_route(car, route[:-1], steps_per_segment=30)

                self.open_exit_barrier()
                self.animate_car(car, route[-2], route[-1], steps=34)
                self.close_exit_barrier()

                car.visible = False

            self.grid[r, c] = 0
            self.car_objects[r][c] = None
            self.update_sensor_lights()

            cars_left += 1

        self.total_left += cars_left
        return cars_left

    def update_metrics(self):
        occupancy_rate = self.get_occupancy_rate()
        self.occupancy_history.append(occupancy_rate)

        if occupancy_rate > self.peak_congestion:
            self.peak_congestion = occupancy_rate

    def show_final_metrics(self):
        average_wait_time = (
            sum(self.wait_times) / len(self.wait_times)
            if self.wait_times
            else 0
        )

        average_occupancy = (
            sum(self.occupancy_history) / len(self.occupancy_history)
            if self.occupancy_history
            else 0
        )

        print("\n" + "=" * 80)
        print("FINAL 3D REALISTIC BASEMENT ML-BASED SIMULATION METRICS")
        print("=" * 80)
        print(f"Total Cars Arrived: {self.total_arrived}")
        print(f"Total Cars Parked: {self.total_parked}")
        print(f"Total Cars Left: {self.total_left}")
        print(f"Total Cars Waiting: {self.total_waiting}")
        print(f"Total Cars Rejected: {self.total_rejected}")
        print(f"Average Wait Time: {average_wait_time:.2f} minutes")
        print(f"Average Occupancy Rate: {average_occupancy * 100:.2f}%")
        print(f"Peak Congestion: {self.peak_congestion * 100:.2f}%")
        print("=" * 80)

    def run(self):
        print("\nSmart Parking 3D Realistic Basement Simulation Started")
        print(f"Parking Basement Size: {ROWS} x {COLS}")
        print(f"Total Slots: {TOTAL_SLOTS}")
        print("Screen-fit centered layout with realistic details, slot sensors, boom barriers, and improved car movement.")

        self.update_sensor_lights()
        self.intro_pause()

        for timestep in range(1, TIME_STEPS + 1):
            hour = timestep % 24

            cars_left = self.remove_cars(hour)

            (
                cars_arrived,
                cars_parked,
                cars_waiting,
                cars_rejected,
                demand_status,
            ) = self.add_cars(hour)

            self.update_metrics()

            print("\n" + "=" * 70)
            print(f"Step: {timestep} | Hour: {hour}:00")
            print(f"ML Prediction: {demand_status}")
            print(f"Cars Arrived: {cars_arrived}")
            print(f"Cars Parked: {cars_parked}")
            print(f"Cars Left: {cars_left}")
            print(f"Cars Waiting: {cars_waiting}")
            print(f"Cars Rejected: {cars_rejected}")
            print(f"Occupancy Rate: {self.get_occupancy_rate() * 100:.2f}%")
            print("=" * 70)

            self.pause_scene(frames=55)

        self.show_final_metrics()

        while True:
            rate(30)


if __name__ == "__main__":
    simulation = RealisticBasementParkingSimulation()
    simulation.run()
