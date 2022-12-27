import random
import numpy as np
import tkinter as tk
import time

base_nucleic_acids = [
    "mN",
    "mE",
    "mS",
    "mW",
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

        return False

    def generate_unoccupied_position(self):
        position = []
        while position == []:
            temp = random.choice(random.choice(self.map))
            if temp[2] == None:
                position = temp

        return (position[0], position[1])


class Blob:
    def __init__(self, acids, coord):
        self.genome = acids
        self.position = coord

    def execute(self):
        # counting the most prevalent genomes
        direction = random.choice(self.genome)
        return direction


def generate_blobs(amount_of_blobs, nucleic_acids):
    blobs = []
    for _ in range(amount_of_blobs):
        genomes = [random.choice(nucleic_acids)
                   for _ in range(amount_of_acids)]
        blob = Blob(genomes, the_map.generate_unoccupied_position())
        blobs.append(blob)
    return blobs


def generate_new_blobs(blobs):
    return


"""
init: make the first generation randomly
"""
amount_of_blobs = 100
amount_of_acids = 32  # nucleic acids make up a genome
map_rows = 48
map_cols = 48
the_map = Map(map_rows, map_cols)
blobs = generate_blobs(amount_of_blobs, base_nucleic_acids)
for blob in blobs:
    the_map.map[blob.position[0]][blob.position[1]][2] = blob

"""
Tkinter setup for visualization
"""
cell_size = 15
window_size = (map_rows * cell_size, map_cols * cell_size)

window = tk.Tk()
window.title("Map Visualization")

canvas = tk.Canvas(window, width=window_size[0], height=window_size[1])
canvas.pack()

# Create a list to store the labels
acid_labels = []

# Create the labels
for i, acid in enumerate(base_nucleic_acids):
    label = tk.Label(window, text=f"{acid}: 0")
    label.pack(side=tk.LEFT)
    acid_labels.append(label)

# Create a label to display the current generation
generation_label = tk.Label(window, text="Generation: 0")
generation_label.pack()


def visualize_map(map: Map):
    canvas.delete("all")

    # Count the occurrences of each nucleic acid in the blobs
    counts = [0] * len(base_nucleic_acids)
    for blob in blobs:
        acid = blob.genome[0]
        index = base_nucleic_acids.index(acid)
        counts[index] += 1

    # Update the labels with the counts
    for i, label in enumerate(acid_labels):
        label.config(text=f"{base_nucleic_acids[i]}: {counts[i]}")

    # Update the generation label
    generation_label.config(text=f"Generation: {generation}")

    for x in range(map.rows):
        for y in range(map.cols):
            cell = map.map[x][y]
            if cell[2] is None:
                canvas.create_rectangle(
                    x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, outline="black")
            else:
                if cell[1] < map.rows / 2:
                    canvas.create_rectangle(
                        x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, fill="green")
                else:
                    canvas.create_rectangle(
                        x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size, fill="black")


old_map = the_map
"""
simulate evolution - reproduction genomes chosen by the popularity of a specific "nucleic acid"
"""
amount_of_generations = 1000
amount_of_moves_per_generation = 50
for generation in range(amount_of_generations):
    for move in range(amount_of_moves_per_generation):
        for blob in blobs:
            direction = blob.execute()
            the_map.move(blob, direction)

    visualize_map(the_map)
    window.update()
    # time.sleep(0.1)

    # starting with the base nucleic acids to not lose them entirely no matter what
    # the higher the comp edge is, the less the nat. selection will have an effect (decimal)
    competetive_edge = 0
    competetive_edge *= (amount_of_blobs * amount_of_acids) / 10
    competetive_edge = round(competetive_edge)
    passing_blobs_acids = base_nucleic_acids * competetive_edge
    print(len(passing_blobs_acids))
    for blob in blobs:
        if blob.position[0] > the_map.rows / 2:
            for genome in blob.genome:
                passing_blobs_acids.append(genome)
    print(len(passing_blobs_acids))

    blobs = generate_blobs(amount_of_blobs, passing_blobs_acids)
    the_map.generate_map()

    for blob in blobs:
        the_map.map[blob.position[0]][blob.position[1]][2] = blob
