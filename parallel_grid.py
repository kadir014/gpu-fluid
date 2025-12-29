from dataclasses import dataclass
from random import uniform
from math import floor, ceil

import pygame


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MAX_FPS = 165

N = 5
DOMAIN = pygame.FRect(0.0, 0.0, 10.0, 10.0)
DOMAIN_TO_SCREEN = 60.0
CELL_SIZE = 5.0
GRID_WIDTH = ceil(DOMAIN.width / CELL_SIZE)
GRID_HEIGHT = ceil(DOMAIN.height / CELL_SIZE)
GRID_CELL_COUNT = GRID_WIDTH * GRID_HEIGHT


pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pygame-CE Basic Template")
clock = pygame.Clock()
is_running = True


@dataclass
class Particle:
    position: pygame.Vector2

@dataclass
class Entry:
    cell_id: int
    particle_id: int

particles: list[Particle] = []
entries: list[Entry] = []
cell_start: list[int] = [0 for _ in range(GRID_CELL_COUNT)]
cell_end: list[int] = [0 for _ in range(GRID_CELL_COUNT)]

for _ in range(N):
    particles.append(Particle(pygame.Vector2(uniform(0, 10), uniform(0, 10))))


while is_running:
    dt = clock.tick(MAX_FPS) * 0.001

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

    mouse = pygame.Vector2(*pygame.mouse.get_pos())
    #particles[0].position = mouse / DOMAIN_TO_SCREEN

    window.fill((255, 255, 255))


    # Pass 1: Convert particles to cell entries
    entries.clear()
    for p in particles:
        cell = p.position / CELL_SIZE
        cell.x = pygame.math.clamp(int(floor(cell.x)), 0, GRID_WIDTH - 1)
        cell.y = pygame.math.clamp(int(floor(cell.y)), 0, GRID_HEIGHT - 1)

        cell_id = int(cell.x + cell.y * GRID_WIDTH)
        entries.append(Entry(cell_id, p))

    # Pass 2: Sort cell entries (Particles belonging to same cell are adjacent)
    #entries.sort(key=lambda x: x.cell_id)


    print("before sorting")
    ids = []
    for entry in entries:
        ids.append(entry.cell_id)
    print(ids)


    # SORTING 1. Histogram
    cell_counts = [0 for _ in range(GRID_CELL_COUNT)]
    for e in entries:
        cell_counts[e.cell_id] += 1
    
    print("cell counts")
    print(cell_counts)

    # SORTING 2. Prefix Sum
    cell_offsets = [0 for _ in range(GRID_CELL_COUNT)]
    cell_offsets[0] = 0
    for i in range(1, GRID_CELL_COUNT):
        cell_offsets[i] = cell_offsets[i - 1] + cell_counts[i - 1]

    print("cell_offsets")
    print(cell_offsets)

    # SORTING 3. Scatter
    sorted_entries = entries.copy()
    for i in range(N):
        cell_id = entries[i].cell_id

        sorted_idx = cell_offsets[cell_id]
        cell_offsets[cell_id] += 1
        
        sorted_entries[sorted_idx] = entries[i]

    entries = sorted_entries.copy()

    print("after sorting")
    ids = []
    for entry in entries:
        ids.append(entry.cell_id)
    print(ids)

    
    
    # SORTING VALIDATION
    for i in range(1, N):
        assert entries[i - 1].cell_id <= entries[i].cell_id


    # Pass 3: Build LUT
    for i in range(N):
        if i == 0 or entries[i].cell_id != entries[i - 1].cell_id:
            cell_start[entries[i].cell_id] = i

        if i == N - 1 or entries[i].cell_id != entries[i + 1].cell_id:
            cell_end[entries[i].cell_id] = i + 1
            
    # Pass 4: Neighbor lookup
    p = mouse / DOMAIN_TO_SCREEN
    cell = p / CELL_SIZE
    cell.x = pygame.math.clamp(int(floor(cell.x)), 0, GRID_WIDTH - 1)
    cell.y = pygame.math.clamp(int(floor(cell.y)), 0, GRID_HEIGHT - 1)
    cell_id = int(cell.x + cell.y * GRID_WIDTH)

    start = cell_start[cell_id]
    end = cell_end[cell_id]
    #print(end - start)


    grid_color = (210, 210, 210)
    for y in range(GRID_HEIGHT + 1):
        line_y = DOMAIN.top * DOMAIN_TO_SCREEN + DOMAIN.height / GRID_HEIGHT * y * DOMAIN_TO_SCREEN
        pygame.draw.line(window, grid_color, (DOMAIN.left * DOMAIN_TO_SCREEN, line_y), (DOMAIN.right * DOMAIN_TO_SCREEN, line_y), 1)

    for x in range(GRID_WIDTH + 1):
        line_x = DOMAIN.left * DOMAIN_TO_SCREEN + DOMAIN.width / GRID_WIDTH * x * DOMAIN_TO_SCREEN
        pygame.draw.line(window, grid_color, (line_x, DOMAIN.top * DOMAIN_TO_SCREEN), (line_x, DOMAIN.bottom * DOMAIN_TO_SCREEN), 1)

    for p in particles:
        pygame.draw.circle(window, (0, 0, 255), p.position * DOMAIN_TO_SCREEN, 5, 0)

    pygame.display.flip()

pygame.quit()