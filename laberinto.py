"""Visualizador de BFS, DFS y A* para el laberinto de Neurito."""

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import random

import pygame


SIZE = 12
CELL = 47
GRID_X, GRID_Y = 30, 132
WIDTH, HEIGHT = 1180, 790
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
STEP_MS = 70

INK = (65, 54, 91)
MUTED = (130, 121, 148)
BG = (250, 244, 253)
PAPER = (255, 253, 255)
WALL = (104, 80, 132)
GRID_LINE = (225, 216, 236)
VISITED = (210, 236, 249)
CURRENT = (255, 179, 150)
PATH = (255, 225, 138)
START_COLOR = (134, 211, 169)
FOOD = (239, 111, 130)
BLUE = (102, 169, 221)
PURPLE = (164, 131, 209)
GREEN = (93, 181, 134)

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ALGORITHMS = ("BFS", "DFS", "A*")
ALGO_COLORS = {"BFS": BLUE, "DFS": PURPLE, "A*": GREEN}
DESCRIPTIONS = {
    "BFS": "Explora por capas con una cola. Encuentra una ruta más corta.",
    "DFS": "Avanza por un camino antes de retroceder. No asegura la ruta más corta.",
    "A*": "Prioriza distancia recorrida + distancia Manhattan restante.",
}


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    explored: tuple[tuple[int, int], ...]
    path: tuple[tuple[int, int], ...]
    max_frontier: int


def neighbors(position: tuple[int, int], maze: tuple[tuple[int, ...], ...]):
    row, col = position
    for dr, dc in DIRECTIONS:
        nxt = (row + dr, col + dc)
        if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and not maze[nxt[0]][nxt[1]]:
            yield nxt


def reconstruct_path(parent, goal):
    if goal not in parent:
        return ()
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return tuple(reversed(path))


def search(maze: tuple[tuple[int, ...], ...], algorithm: str) -> SearchResult:
    """Run one of the three searches; every move has unit cost."""
    parent = {START: None}
    explored = []

    if algorithm == "BFS":
        frontier = deque([START])
        discovered = {START}
        max_frontier = 1
        while frontier:
            node = frontier.popleft()
            explored.append(node)
            if node == GOAL:
                break
            for nxt in neighbors(node, maze):
                if nxt not in discovered:
                    discovered.add(nxt)
                    parent[nxt] = node
                    frontier.append(nxt)
            max_frontier = max(max_frontier, len(frontier))
    elif algorithm == "DFS":
        frontier = [START]
        discovered = {START}
        max_frontier = 1
        while frontier:
            node = frontier.pop()
            explored.append(node)
            if node == GOAL:
                break
            next_positions = list(neighbors(node, maze))
            for nxt in reversed(next_positions):
                if nxt not in discovered:
                    discovered.add(nxt)
                    parent[nxt] = node
                    frontier.append(nxt)
            max_frontier = max(max_frontier, len(frontier))
    elif algorithm == "A*":
        sequence = itertools.count()
        frontier = [(abs(START[0] - GOAL[0]) + abs(START[1] - GOAL[1]), next(sequence), 0, START)]
        g_score = {START: 0}
        closed = set()
        max_frontier = 1
        while frontier:
            _, _, cost, node = heapq.heappop(frontier)
            if node in closed or cost != g_score[node]:
                continue
            closed.add(node)
            explored.append(node)
            if node == GOAL:
                break
            for nxt in neighbors(node, maze):
                new_cost = cost + 1
                if new_cost < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_cost
                    parent[nxt] = node
                    heuristic = abs(nxt[0] - GOAL[0]) + abs(nxt[1] - GOAL[1])
                    heapq.heappush(frontier, (new_cost + heuristic, next(sequence), new_cost, nxt))
            max_frontier = max(max_frontier, len(frontier))
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")

    return SearchResult(algorithm, tuple(explored), reconstruct_path(parent, GOAL), max_frontier)


def make_maze(seed: int = 16319) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Create a fixed, solvable 12x12 maze with a reproducible seed."""
    for candidate_seed in itertools.count(seed):
        rng = random.Random(candidate_seed)
        cells = [[int(rng.random() < 0.25) for _ in range(SIZE)] for _ in range(SIZE)]
        cells[START[0]][START[1]] = 0
        cells[GOAL[0]][GOAL[1]] = 0
        maze = tuple(tuple(row) for row in cells)
        if search(maze, "BFS").path:
            return maze, candidate_seed


MAZE, MAZE_SEED = make_maze()


class LaberintoVisual:
    """Pygame scene that animates search expansion and Neurito's final route."""

    def __init__(self, heading_font, body_font, small_font, tiny_font):
        self.heading_font = heading_font
        self.body_font = body_font
        self.small_font = small_font
        self.tiny_font = tiny_font
        self.algorithm = "A*"
        self.result = search(MAZE, self.algorithm)
        self.expanded_count = 0
        self.path_count = 0
        self.playing = True
        self.accumulator = 0
        self.back_rect = pygame.Rect(868, 724, 252, 42)
        self.buttons = {
            algorithm: pygame.Rect(666 + index * 148, 143, 132, 46)
            for index, algorithm in enumerate(ALGORITHMS)
        }
        self.restart()

    def restart(self):
        self.result = search(MAZE, self.algorithm)
        self.expanded_count = 0
        self.path_count = 0
        self.accumulator = 0
        self.playing = True

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                return "home"
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.algorithm = ALGORITHMS[event.key - pygame.K_1]
                self.restart()
            elif event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.key == pygame.K_r:
                self.restart()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.back_rect.collidepoint(event.pos):
                return "home"
            for algorithm, rect in self.buttons.items():
                if rect.collidepoint(event.pos):
                    self.algorithm = algorithm
                    self.restart()
                    break
        return None

    def update(self, elapsed_ms: int):
        if not self.playing:
            return
        self.accumulator += elapsed_ms
        while self.accumulator >= STEP_MS:
            self.accumulator -= STEP_MS
            if self.expanded_count < len(self.result.explored):
                self.expanded_count += 1
            elif self.path_count < len(self.result.path):
                self.path_count += 1
            else:
                self.playing = False

    def _draw_text(self, surface, font, text, color, pos):
        surface.blit(font.render(text, True, color), pos)

    def draw(self, surface):
        surface.fill(BG)
        self._draw_text(surface, self.heading_font, "Neurito busca su comida", INK, (30, 25))
        self._draw_text(surface, self.small_font,
                        "Laberinto de 12 × 12 · observa cómo trabaja cada algoritmo",
                        MUTED, (33, 62))

        grid_rect = pygame.Rect(GRID_X - 10, GRID_Y - 10, SIZE * CELL + 20, SIZE * CELL + 20)
        pygame.draw.rect(surface, PAPER, grid_rect, border_radius=18)
        pygame.draw.rect(surface, (234, 224, 241), grid_rect, width=2, border_radius=18)

        explored_set = set(self.result.explored[:self.expanded_count])
        current = self.result.explored[self.expanded_count - 1] if self.expanded_count else None
        revealed_path = self.result.path[:self.path_count]
        revealed_path_set = set(revealed_path)

        for row in range(SIZE):
            for col in range(SIZE):
                pos = (row, col)
                rect = pygame.Rect(GRID_X + col * CELL, GRID_Y + row * CELL, CELL, CELL)
                if MAZE[row][col]:
                    color = WALL
                elif pos in revealed_path_set:
                    color = PATH
                elif pos == current:
                    color = CURRENT
                elif pos in explored_set:
                    color = VISITED
                elif pos == START:
                    color = START_COLOR
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, GRID_LINE, rect, width=1)

                if pos == START and not revealed_path:
                    self._draw_text(surface, self.heading_font, "N", INK, (rect.x + 15, rect.y + 8))
                if pos == GOAL:
                    apple_center = (rect.right - 12, rect.y + 13)
                    pygame.draw.circle(surface, FOOD, apple_center, 8)
                    pygame.draw.ellipse(surface, START_COLOR,
                                        (apple_center[0] + 1, apple_center[1] - 11, 8, 4))

        # Neurito follows the route only after the search has finished expanding.
        if revealed_path:
            mouse_cell = revealed_path[-1]
        elif current is not None:
            mouse_cell = current
        else:
            mouse_cell = START
        row, col = mouse_cell
        center = (GRID_X + col * CELL + CELL // 2, GRID_Y + row * CELL + CELL // 2)
        pygame.draw.circle(surface, (87, 179, 133), center, 14)
        pygame.draw.circle(surface, (255, 255, 255), (center[0] - 5, center[1] - 3), 2)
        pygame.draw.circle(surface, (255, 255, 255), (center[0] + 5, center[1] - 3), 2)
        pygame.draw.arc(surface, INK, (center[0] - 6, center[1] - 1, 12, 9), 0.2, 2.9, 2)

        panel = pygame.Rect(650, 112, 500, 590)
        pygame.draw.rect(surface, PAPER, panel, border_radius=22)
        pygame.draw.rect(surface, (234, 224, 241), panel, width=2, border_radius=22)
        self._draw_text(surface, self.heading_font, "Elige una búsqueda", INK, (674, 119))
        for algorithm, rect in self.buttons.items():
            color = ALGO_COLORS[algorithm]
            pygame.draw.rect(surface, color if algorithm == self.algorithm else (244, 239, 248),
                             rect, border_radius=14)
            self._draw_text(surface, self.body_font, algorithm,
                            (255, 255, 255) if algorithm == self.algorithm else INK,
                            (rect.x + 48, rect.y + 12))
        self._draw_text(surface, self.small_font, DESCRIPTIONS[self.algorithm], MUTED, (674, 207))

        expanded_total = len(self.result.explored)
        route_length = max(0, len(self.result.path) - 1)
        self._draw_text(surface, self.body_font,
                        f"Celdas expandidas: {self.expanded_count} / {expanded_total}", INK, (674, 267))
        self._draw_text(surface, self.body_font,
                        f"Longitud de la ruta: {route_length} pasos", INK, (674, 301))
        self._draw_text(surface, self.body_font,
                        f"Frontera máxima: {self.result.max_frontier} celdas", INK, (674, 335))
        self._draw_text(surface, self.small_font,
                        f"Laberinto fijo · semilla {MAZE_SEED} · movimientos de costo 1", MUTED, (674, 375))

        pygame.draw.line(surface, (232, 224, 238), (674, 410), (1125, 410), 2)
        self._draw_text(surface, self.heading_font, "¿Cuál conviene más?", INK, (674, 430))
        bfs_result = search(MAZE, "BFS")
        dfs_result = search(MAZE, "DFS")
        astar_result = search(MAZE, "A*")
        self._draw_text(surface, self.small_font,
                        f"A*: {len(astar_result.path) - 1} pasos, {len(astar_result.explored)} celdas.",
                        MUTED, (674, 468))
        self._draw_text(surface, self.small_font,
                        f"BFS: {len(bfs_result.path) - 1} pasos, {len(bfs_result.explored)} celdas.",
                        MUTED, (674, 493))
        self._draw_text(surface, self.small_font,
                        f"DFS: {len(dfs_result.path) - 1} pasos, {len(dfs_result.explored)} celdas.",
                        MUTED, (674, 518))
        self._draw_text(surface, self.small_font,
                        "Mejor aquí: A*, ruta óptima con menos expansiones.", INK, (674, 543))

        self._draw_text(surface, self.small_font,
                        "Espacio: pausar / continuar · R: reiniciar · 1/2/3: elegir",
                        MUTED, (674, 602))
        self._draw_text(surface, self.small_font,
                        "BFS y A* son óptimos con pasos de igual costo.",
                        MUTED, (674, 629))
        pygame.draw.rect(surface, (238, 229, 255), self.back_rect, border_radius=15)
        self._draw_text(surface, self.body_font, "← Volver al laboratorio", INK,
                        (self.back_rect.x + 45, self.back_rect.y + 10))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neurito · Laberinto de búsqueda")
    clock = pygame.time.Clock()
    heading = pygame.font.SysFont("Arial", 23, bold=True)
    body = pygame.font.SysFont("Arial", 18)
    small = pygame.font.SysFont("Arial", 15)
    tiny = pygame.font.SysFont("Arial", 13)
    scene = LaberintoVisual(heading, body, small, tiny)
    running = True
    while running:
        elapsed = clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                scene.handle_event(event)
        scene.update(elapsed)
        scene.draw(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
