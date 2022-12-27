import pygame
import random
import numpy as np
import os
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

    def execute(self, blob, map: Map):
        # counting the most prevalent genomes
        direction = random.choice(self.genome)
        map.move(blob, direction)


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
amount_of_acids = 4  # nucleic acids make up a genome
map_rows = 32
map_cols = 32
the_map = Map(map_rows, map_cols)
blobs = generate_blobs(amount_of_blobs, base_nucleic_acids)
for blob in blobs:
    the_map.map[blob.position[0]][blob.position[1]][2] = blob

"""
set up pygame to visualize data
"""
# Set the size of the window and the size of each cell on the map
cell_size = 40
window_size = (map_rows*cell_size, map_cols*cell_size)

# Initialize Pygame and create a window
pygame.init()
window = pygame.display.set_mode(window_size)

# Set the title of the window
pygame.display.set_caption("Map Visualization")

# Create a font for drawing text on the window
font = pygame.font.Font(None, 36)


def visualize_map(map: Map):
    # Clear the window
    window.fill((255, 255, 255))

    # Iterate through the cells on the map and draw them on the window
    for x in range(map.rows):
        for y in range(map.cols):
            cell = map.map[x][y]
            if cell[2] is None:
                if cell[0] > map.rows/5:
                    # Draw a cell with a blob in the passing_blobs range as a green rectangle
                    pygame.draw.rect(
                        window, (0, 255, 0), (x*cell_size, y*cell_size, cell_size, cell_size), 0)
                else:
                    # Draw an empty cell
                    pygame.draw.rect(
                        window, (0, 0, 0), (x*cell_size, y*cell_size, cell_size, cell_size), 1)
            else:
                # Draw a cell with a blob as a black rectangle
                pygame.draw.rect(
                    window, (0, 0, 0), (x*cell_size, y*cell_size, cell_size, cell_size), 0)

    # Update the window to show the new map
    pygame.display.flip()


"""
simulate evolution - reproduction genomes chosen by the popularity of a specific "nucleic acid"
"""
amount_of_generations = 1000
amount_of_moves_per_generation = 10000
for generation in range(amount_of_generations):
    for move in range(amount_of_moves_per_generation):
        for blob in blobs:
            blob.execute(blob, the_map)

    passing_blobs_acids = []
    for blob in blobs:
        if blob.position[0] > the_map.rows/5:
            for genome in blob.genome:
                passing_blobs_acids.append(genome)

    blobs = generate_blobs(amount_of_blobs, passing_blobs_acids)
    the_map.generate_map()

    for blob in blobs:
        the_map.map[blob.position[0]][blob.position[1]][2] = blob

    visualize_map(the_map)
