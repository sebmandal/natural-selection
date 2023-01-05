import random
import math
import tkinter as tk

base_neurons = [
    "mN",
    "mN",
    "mE",
    "mE",
    "mS",
    "mS",
    "mW",
    "mW",
    "sG",
    "sM"
]


class Map:
    def __init__(self, rows, cols):
        self.map = []
        self.rows = rows
        self.cols = cols
        self.generate_map()

    def generate_map(self):
        self.map = [[[x, y, None]
                    for x in range(self.rows)] for y in range(self.cols)]
        return

    def count_occupied_neighbors(self, blob):
        x, y = blob.position
        occupied_count = 0

        # Check the cells in a 3x3 area around the given position
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i == x and j == y:  # cur cell
                    continue
                if the_map.map[i][j][2] is not None:
                    occupied_count += 1

        return occupied_count

    def move(self, blob, direction):
        """
        check coordinates in any direction provided
        if you can move there, change its state to the current blob and turn the blobs current position to None
        state can be: None (unoccupied)
        or: blob (the blob instance)
        """

        x, y = blob.position[0], blob.position[1]

        if direction == "mN":
            if y > 0 and self.map[x][y-1][2] is None:
                self.map[x][y][2] = None
                self.map[x][y-1][2] = blob
                blob.position = (x, y-1)
                return True

        if direction == "mE":
            if x < len(self.map[0])-1 and self.map[x+1][y][2] is None:
                self.map[x][y][2] = None
                self.map[x+1][y][2] = blob
                blob.position = (x+1, y)
                return True

        if direction == "mS":
            if y < len(self.map)-1 and self.map[x][y+1][2] is None:
                self.map[x][y][2] = None
                self.map[x][y+1][2] = blob
                blob.position = (x, y+1)
                return True

        if direction == "mW":
            if x > 0 and self.map[x-1][y][2] is None:
                self.map[x][y][2] = None
                self.map[x-1][y][2] = blob
                blob.position = (x-1, y)
                return True

        if direction == "sG":
            """
            gravitational pull based on all other blobs using the distance to them using pythagoras, it moves in the closest direction to the pull, if the pull is taken, just stay where you are
            """

            # Find all blobs on the map
            all_blobs = [
                blob for row in the_map.map for blob in row if blob[2] is not None]

            # Calculate the distance to each blob using Pythagoras
            distances = []
            for other_blob in all_blobs:
                x_diff = other_blob[0] - blob.position[0]
                y_diff = other_blob[1] - blob.position[1]
                distance = math.sqrt(x_diff**2 + y_diff**2)
                distances.append(distance)

            # Find the blob with the minimum distance
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            nearest_blob = all_blobs[min_index]

            # Calculate the direction to the nearest blob
            x_diff = nearest_blob[0] - blob.position[0]
            y_diff = nearest_blob[1] - blob.position[1]

            # If the nearest blob is to the north or south, move in that direction
            if y_diff < 0:
                direction = "mN"
            elif y_diff > 0:
                direction = "mS"
            # If the nearest blob is to the east or west, move in that direction
            elif x_diff < 0:
                direction = "mW"
            elif x_diff > 0:
                direction = "mE"
            # If the blob is already at the same position as the nearest blob, do not move
            else:
                direction = None

            # If a valid direction was found, move in that direction
            if direction is not None:
                the_map.move(blob, direction)

        if direction == "sM":
            """
            the cell moves toward the middle
            """
            # Calculate the direction to the center of the map
            center_x = the_map.cols // 2
            center_y = the_map.rows // 2
            x_diff = center_x - blob.position[0]
            y_diff = center_y - blob.position[1]

            # If the center is to the north or south, move in that direction
            if abs(y_diff) > abs(x_diff):
                if y_diff < 0:
                    direction = "mN"
                elif y_diff > 0:
                    direction = "mS"
            # If the center is to the east or west, move in that direction
            else:
                if x_diff < 0:
                    direction = "mW"
                elif x_diff > 0:
                    direction = "mE"
                else:
                    direction = None

            # If a valid direction was found, move in that direction
            if direction is not None:
                the_map.move(blob, direction)

        return False

    def generate_unoccupied_position(self):
        position = []
        while position == []:
            temp = random.choice(random.choice(self.map))
            if temp[2] == None:
                position = temp

        return (position[0], position[1])


class Blob:
    def __init__(self, neurons, coord):
        self.genome = neurons
        self.position = coord

    def execute(self):
        # counting the most prevalent genomes
        direction = random.choice(self.genome)
        return direction


def generate_blobs(amount_of_blobs, neurons):
    blobs = []
    for _ in range(amount_of_blobs):
        genomes = [random.choice(neurons)
                   for _ in range(amount_of_neurons)]
        blob = Blob(genomes, the_map.generate_unoccupied_position())
        blobs.append(blob)
    return blobs


def generate_new_blobs(blobs):
    return


"""
init: make the first generation randomly
"""
map_rows = 32
map_cols = map_rows
the_map = Map(map_rows, map_cols)
amount_of_blobs = map_rows * 2
amount_of_neurons = 10000  # neurons make up a genome
blobs = generate_blobs(amount_of_blobs, base_neurons)
for blob in blobs:
    the_map.map[blob.position[0]][blob.position[1]][2] = blob

"""
Tkinter setup for visualization
"""
cell_size = map_rows // 3
window_size = (map_rows * cell_size, map_cols * cell_size)

window = tk.Tk()
window.title("Map Visualization")

canvas = tk.Canvas(window, width=window_size[0], height=window_size[1])
canvas.pack(side=tk.BOTTOM)

# Create a list to store the labels
neuron_labels = []
neuron_packed = []

# Create the labels
for i, neuron in enumerate(base_neurons):
    if neuron not in neuron_packed:
        neuron_packed.append(neuron)
        label = tk.Label(window, text=f"{neuron}: 0")
        label.pack(side=tk.LEFT)
        neuron_labels.append(label)

# Create a label to display the current generation
generation_label = tk.Label(window, text="Generation: 0")
generation_label.pack(side=tk.RIGHT)
generation_label.pack()


def visualize_map(map: Map):
    canvas.delete("all")

    # Count the occurrences of each neuron in the blobs
    counts = [0] * len(base_neurons)
    for blob in blobs:
        neuron = blob.genome[0]
        index = base_neurons.index(neuron)
        counts[index] += 1

    # Update the labels with the counts
    for i, label in enumerate(neuron_labels):
        label.config(text=f"{base_neurons[i]}: {counts[i]}")

    # Update the generation label
    generation_label.config(text=f"Generation: {generation}")

    for x in range(map.rows):
        for y in range(map.cols):
            cell = map.map[x][y]
            if cell[2] is None:
                canvas.create_rectangle(
                    x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, fill="black")
            else:
                if cell[1] > the_map.cols - 3:
                    canvas.create_rectangle(
                        x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, fill="green")
                else:
                    canvas.create_rectangle(
                        x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, fill="white")

    window.update()


"""
simulate evolution - reproduction genomes chosen by the popularity of a specific "neuron"
"""
amount_of_generations = 100000000
amount_of_moves_per_generation = 50
for generation in range(amount_of_generations):
    for move in range(amount_of_moves_per_generation):
        for blob in blobs:
            direction = blob.execute()
            the_map.move(blob, direction)

    visualize_map(the_map)

    # starting with the base neurons to not lose them entirely no matter what
    # the higher the comp edge is, the less the nat. selection will have an effect (decimal)
    competetive_edge = 0.00001
    competetive_edge *= (amount_of_blobs * amount_of_neurons) // 10
    competetive_edge = round(competetive_edge)
    passing_blobs_neurons = base_neurons * competetive_edge
    for blob in blobs:
        """
        MIDDLE:
        center_x = the_map.cols // 2
        center_y = the_map.rows // 2
        x_diff = center_x - blob.position[0]
        y_diff = center_y - blob.position[1]
        distance = math.sqrt(x_diff**2 + y_diff**2)
        if distance <= 5 and the_map.count_occupied_neighbors(blob) > 5:
        """
        if blob.position[1] > the_map.cols - 3:
            for genome in blob.genome:
                passing_blobs_neurons.append(genome)

    if len(passing_blobs_neurons) == 0:
        passing_blobs_neurons = base_neurons

    blobs = generate_blobs(amount_of_blobs, passing_blobs_neurons)
    the_map.generate_map()

    for blob in blobs:
        the_map.map[blob.position[0]][blob.position[1]][2] = blob
