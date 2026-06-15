# Phase 3 Simulation Explanation

Phase 3 focuses on the live 3D parking simulation.

## Main Idea

The simulation shows how cars enter a parking lot, select an available slot, park, and later exit. It converts machine learning prediction and parking logic into a visual demo.

## Slot Selection Logic

The system first checks all parking slots and creates a list of available slots. A car is then assigned to one available slot using JavaScript random selection logic.

The car never parks in an occupied slot because only free slots are selected.

## Sensor Logic

- Green sensor means the slot is available.
- Red sensor means the slot is occupied.

When a car parks, the sensor turns red. When the car exits, the sensor turns green again.

## Demand Labels

Demand is based on occupancy percentage:

- Low Demand: 0% to 34%
- Medium Demand: 35% to 69%
- High Demand: 70% to 100%

## Libraries Used

### Three.js
Used for creating and rendering the 3D parking lot in the browser.

### JavaScript
Used for simulation logic such as car movement, random slot selection, sensor updates, and metrics.

### OrbitControls
Used for 360-degree camera rotation, zoom, and panning.

### Vite
Used to build and deploy the web-based simulation.

### Streamlit iframe
Used to embed the Vercel simulation inside the Streamlit dashboard.
