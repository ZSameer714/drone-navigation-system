import math
from typing import List, Tuple, Dict
import heapq
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

class UnionFind:
    def __init__(self, size: int):  # ✅ Correct!
        self.parent = list(range(size))
        self.rank = [0] * size


    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return False
            
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
        return True

class DroneNavigation:
    def __init__(self):
        self.stations: List[Tuple[float, float]] = []
        self.edges: List[Tuple[float, int, int]] = []
        self.mst_edges: List[Tuple[int, int, float]] = []
        self.adjacency_list: Dict[int, List[Tuple[int, float]]] = {}
        self.construction_steps: List[List[Tuple[int, int, float]]] = []

    def is_valid_station_position(self, x: float, y: float) -> bool:
        # Check for NaN or infinity
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            print("❌ Error: Invalid coordinates (NaN or infinity)")
            return False
        
        # Check for reasonable coordinate range
        if abs(x) > 10000 or abs(y) > 10000:
            print("❌ Error: Coordinates are too large. Please use values between -10000 and 10000")
            return False
        
        return True

    def add_station(self, x: float, y: float) -> bool:
        if not self.is_valid_station_position(x, y):
            return False

        # Check for duplicate stations with a small tolerance
        for station_x, station_y in self.stations:
            distance = math.sqrt((x - station_x)*2 + (y - station_y)*2)
            if distance < 0.1:  # Increased tolerance to 0.1 units
                print("❌ Error: A station already exists at or very close to these coordinates")
                print(f"    Existing station: ({station_x}, {station_y})")
                print(f"    New station: ({x}, {y})")
                print(f"    Distance between stations: {distance:.4f} units")
                return False

        self.stations.append((x, y))
        self.adjacency_list[len(self.stations) - 1] = []
        return True

    def calculate_distances(self) -> None:
        self.edges = []  # Clear existing edges
        n = len(self.stations)
        for i in range(n):
            self.adjacency_list[i] = []  # Clear existing adjacency list
            for j in range(i + 1, n):
                x1, y1 = self.stations[i]
                x2, y2 = self.stations[j]
                distance = math.sqrt((x2 - x1)*2 + (y2 - y1)*2)
                if distance > 0.1:  # Only add edges with significant distance
                    self.edges.append((distance, i, j))
                    self.adjacency_list[i].append((j, distance))
                    self.adjacency_list[j].append((i, distance))

    def find_minimum_spanning_tree_kruskal(self) -> float:
        if not self.edges:
            self.calculate_distances()
        
        if len(self.stations) < 2:
            raise ValueError("Need at least 2 different stations to create a MST")
            
        if not self.edges:
            raise ValueError("No valid edges found between stations. All stations might be too close to each other.")
            
        # Sort edges by distance
        self.edges.sort()
        uf = UnionFind(len(self.stations))
        total_distance = 0.0
        self.mst_edges = []
        self.construction_steps = []
        
        for distance, u, v in self.edges:
            if uf.union(u, v):
                self.mst_edges.append((u, v, distance))
                total_distance += distance
                self.construction_steps.append(self.mst_edges.copy())
                
        # Check if we have a valid MST (n-1 edges for n vertices)
        if len(self.mst_edges) != len(self.stations) - 1:
            error_msg = "Could not construct a valid MST. "
            if not self.mst_edges:
                error_msg += "No edges were added to the MST. "
            else:
                error_msg += f"Only {len(self.mst_edges)} edges were added (need {len(self.stations) - 1}). "
            error_msg += "Check if stations are properly spaced."
            raise ValueError(error_msg)
                
        return total_distance

    def find_minimum_spanning_tree_prim(self) -> float:
        if not self.adjacency_list:
            self.calculate_distances()
            
        if len(self.stations) < 2:
            raise ValueError("Need at least 2 different stations to create a MST")
            
        if not self.edges:
            raise ValueError("No valid edges found between stations. All stations might be too close to each other.")
        
        n = len(self.stations)
        visited = [False] * n
        min_heap = []
        total_distance = 0.0
        self.mst_edges = []
        self.construction_steps = []
        
        heapq.heappush(min_heap, (0.0, 0, -1))  
        
        while min_heap and len(self.mst_edges) < n - 1:
            dist, u, parent = heapq.heappop(min_heap)
            
            if visited[u]:
                continue
                
            visited[u] = True
            if parent != -1:
                self.mst_edges.append((parent, u, dist))
                total_distance += dist
                self.construction_steps.append(self.mst_edges.copy())
                
            for v, weight in self.adjacency_list[u]:
                if not visited[v]:
                    heapq.heappush(min_heap, (weight, v, u))
        
        # Check if we have a valid MST (n-1 edges for n vertices)
        if len(self.mst_edges) != len(self.stations) - 1:
            error_msg = "Could not construct a valid MST. "
            if not self.mst_edges:
                error_msg += "No edges were added to the MST. "
            else:
                error_msg += f"Only {len(self.mst_edges)} edges were added (need {len(self.stations) - 1}). "
            error_msg += "Check if stations are properly spaced."
            raise ValueError(error_msg)
                    
        return total_distance

    def get_delivery_path(self) -> List[Tuple[int, int, float]]:
        return self.mst_edges

    def visualize(self, algorithm_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot stations
        x_coords = [x for x, _ in self.stations]
        y_coords = [y for _, y in self.stations]
        
        # Plot all stations
        stations_scatter = ax.scatter(x_coords, y_coords, color='blue', s=100, label='Delivery Stations')
        
        # Add station numbers
        station_labels = []
        for i, (x, y) in enumerate(self.stations):
            label = ax.annotate(f'Station {i+1}', (x, y), xytext=(5, 5), 
                              textcoords='offset points')
            station_labels.append(label)
        
        # Create a colormap for the edges
        cmap = LinearSegmentedColormap.from_list('edge_cmap', ['red', 'green'])
        
        # Initialize edge lines
        edge_lines = []
        edge_labels = []
        
        def init():
            ax.set_title(f'Drone Delivery Network - {algorithm_name} Algorithm\n'
                        f'Construction of Minimum Spanning Tree')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True)
            ax.legend()
            return stations_scatter, *station_labels, *edge_lines, *edge_labels
        
        def update(frame):
            # Clear previous edges
            for line in edge_lines:
                line.remove()
            edge_lines.clear()
            
            for label in edge_labels:
                label.remove()
            edge_labels.clear()
            
            # Plot current edges
            current_edges = self.construction_steps[frame]
            for u, v, dist in current_edges:
                x1, y1 = self.stations[u]
                x2, y2 = self.stations[v]
                
                # Calculate color based on frame
                color = cmap(frame / len(self.construction_steps))
                
                line = ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)[0]
                edge_lines.append(line)
                
                # Add distance label
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                label = ax.annotate(f'{dist:.2f}', (mid_x, mid_y), xytext=(0, 5),
                                  textcoords='offset points', ha='center')
                edge_labels.append(label)
            
            # Update title with current progress
            total_distance = sum(dist for _, _, dist in current_edges)
            ax.set_title(f'Drone Delivery Network - {algorithm_name} Algorithm\n'
                        f'Step {frame + 1}/{len(self.construction_steps)} | '
                        f'Current Distance: {total_distance:.2f} units')
            
            return stations_scatter, *station_labels, *edge_lines, *edge_labels
        
        
        anim = FuncAnimation(fig, update, frames=len(self.construction_steps),
                           init_func=init, blit=True, interval=1000)
        
        plt.tight_layout()
        plt.show()

def get_valid_number(prompt: str, min_value: float = None) -> float:
    while True:
        try:
            value = float(input(prompt))
            if min_value is not None and value < min_value:
                print(f"❌ Error: Value must be at least {min_value}")
                continue
            return value
        except ValueError:
            print("❌ Error: Please enter a valid number")

def get_valid_integer(prompt: str, min_value: int = None, max_value: int = None) -> int:
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"❌ Error: Value must be at least {min_value}")
                continue
            if max_value is not None and value > max_value:
                print(f"❌ Error: Value must be at most {max_value}")
                continue
            return value
        except ValueError:
            print("❌ Error: Please enter a valid integer")

def print_station_info(stations: List[Tuple[float, float]]) -> None:
    print("\n📌 Station Information:")
    for i, (x, y) in enumerate(stations, 1):
        print(f"Station {i}: Coordinates ({x:.2f}, {y:.2f})")

def print_mst_edges(edges: List[Tuple[int, int, float]]) -> None:
    print("\n🛣 Minimum Spanning Tree Edges:")
    for u, v, dist in edges:
        print(f"Station {u+1} -- Station {v+1} | Distance: {dist:.2f}")

def read_input_from_file(filename: str) -> List[Tuple[float, float]]:
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            n = int(lines[0].strip())
            stations = []
            for line in lines[1:n+1]:
                x, y = map(float, line.strip().split())
                stations.append((x, y))
            return stations
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: Invalid data format in file: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        sys.exit(1)

def main():
    print("🚁 Drone Navigation System Using Minimum Spanning Tree")
    print("=" * 50)
    
    nav = DroneNavigation()
    
    
    if len(sys.argv) > 1:
        print(f"📂 Reading input from file: {sys.argv[1]}")
        stations = read_input_from_file(sys.argv[1])
        for x, y in stations:
            if not nav.add_station(x, y):
                print(f"❌ Warning: Skipping duplicate station at ({x:.2f}, {y:.2f})")
    else:
    
        while True:
            try:
                n = get_valid_integer("Enter number of delivery stations: ", min_value=2)
                break
            except ValueError as e:
                print(f"❌ Error: {str(e)}")
        
        
        stations_added = 0
        while stations_added < n:
            print(f"\nEnter coordinates for Station {stations_added + 1}:")
            try:
                x = get_valid_number("X coordinate: ")
                y = get_valid_number("Y coordinate: ")
                if nav.add_station(x, y):
                    stations_added += 1
            except ValueError as e:
                print(f"❌ Error: {str(e)}")
    
    if len(nav.stations) < 2:
        print("❌ Error: Need at least 2 different stations to create a MST")
        return
    
    
    print_station_info(nav.stations)
    
    
    print("\nChoose algorithm:")
    print("1. Kruskal's Algorithm")
    print("2. Prim's Algorithm")
    
    while True:
        try:
            choice = get_valid_integer("Enter your choice (1 or 2): ", min_value=1, max_value=2)
            break
        except ValueError as e:
            print(f"❌ Error: {str(e)}")
    
    try:
        
        if choice == 1:
            total_distance = nav.find_minimum_spanning_tree_kruskal()
            algorithm_name = "Kruskal's"
            print("\nUsing Kruskal's Algorithm...")
        else:
            total_distance = nav.find_minimum_spanning_tree_prim()
            algorithm_name = "Prim's"
            print("\nUsing Prim's Algorithm...")
        
        
        path = nav.get_delivery_path()
        
        print("\n📊 Results:")
        print(f"Total minimum distance: {total_distance:.2f} units")
        print_mst_edges(path)
        
        
        print("\n🖼 Generating visualization...")
        nav.visualize(algorithm_name)
        
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")