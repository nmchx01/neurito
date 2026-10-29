"""Neurito: a tiny, friendly neural-network playground."""

from pathlib import Path

import numpy as np
import pygame


WIDTH, HEIGHT = 1150, 780
ACTIONS = ("comer", "dormir", "jugar")
ACTION_LABELS = {"comer": "COMER", "dormir": "DORMIR", "jugar": "JUGAR", "calma": "MODO CALMA"}
ACTION_COLORS = {"comer": (255, 162, 104), "dormir": (139, 164, 255), "jugar": (109, 205, 157)}
NEED_LABELS = ("Hambre", "Energía", "Aburrimiento")
NEED_COLORS = ((255, 149, 109), (115, 191, 241), (166, 139, 245))
INK = (65, 54, 91)
MUTED = (130, 121, 148)
PAPER = (255, 253, 255)
BG = (250, 244, 253)
PINK = (250, 224, 241)
LILAC = (238, 229, 255)
MINT = (226, 248, 235)
GOLD = (255, 242, 201)


def relu(values: np.ndarray) -> np.ndarray:
    """Keep positive activations and turn negative ones into zero."""
    return np.maximum(0, values)


def softmax(values: np.ndarray) -> np.ndarray:
    """Turn action scores into probabilities that add up to one."""
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    train = rng.random((3000, 3))
    train_y = np.argmax(np.column_stack((train[:, 0], 1 - train[:, 1], train[:, 2])), axis=1)
    return train, train_y


def train_network(
    train: np.ndarray,
    targets: np.ndarray,
    hidden_size: int = 6,
    epochs: int = 1200,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Train a small two-layer network on the toy policy."""
    rng = np.random.default_rng(7)
    w1 = rng.standard_normal((3, hidden_size)) * 0.5
    b1 = np.zeros((1, hidden_size))
    w2 = rng.standard_normal((hidden_size, 3)) * 0.5
    b2 = np.zeros((1, 3))

    for _ in range(epochs):
        z1 = train @ w1 + b1
        hidden = relu(z1)
        probabilities = softmax(hidden @ w2 + b2)
        delta2 = probabilities.copy()
        delta2[np.arange(len(train)), targets] -= 1
        delta2 /= len(train)
        grad_w2 = hidden.T @ delta2
        grad_b2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ w2.T) * (z1 > 0)
        grad_w1 = train.T @ delta1
        grad_b1 = np.sum(delta1, axis=0, keepdims=True)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    return w1, b1, w2, b2, float(epochs), float(learning_rate)


def probabilities_for(needs: list[float], model: tuple) -> np.ndarray:
    w1, b1, w2, b2, _, _ = model
    hidden = relu(np.asarray([needs]) @ w1 + b1)
    return softmax(hidden @ w2 + b2)[0]


def draw_card(surface: pygame.Surface, rect: pygame.Rect, color=PAPER, radius=24) -> None:
    pygame.draw.rect(surface, (232, 222, 239), rect.move(0, 5), border_radius=radius)
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, (239, 230, 244), rect, width=2, border_radius=radius)


def draw_text(surface, font, text, color, pos, *, center=False) -> pygame.Rect:
    image = font.render(text, True, color)
    rect = image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
    surface.blit(image, rect)
    return rect


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neurito · Mi cerebro juguetón")
    clock = pygame.time.Clock()

    # Comic Sans is a friendly first choice on Windows; the other round fonts
    # keep the interface readable on computers where it is not installed.
    font_names = ("Comic Sans MS", "Comic Sans", "Trebuchet MS", "Arial Rounded MT Bold", "Arial")
    font_name = next((name for name in font_names if pygame.font.match_font(name)), None)
    title_font = pygame.font.SysFont(font_name, 31, bold=True)
    heading_font = pygame.font.SysFont(font_name, 21, bold=True)
    body_font = pygame.font.SysFont(font_name, 18)
    small_font = pygame.font.SysFont(font_name, 15)
    tiny_font = pygame.font.SysFont(font_name, 13)

    train, train_y = make_dataset()
    model = train_network(train, train_y)

    asset_dir = Path(__file__).resolve().parent / "assets"
    pet_size = (264, 264)
    frame_files = {
        "calma": ("neutral_1.png", "neutral_2.png", "neutral_3.png"),
        "comer": ("comer_1.png", "comer_2.png"),
        "dormir": ("dormir_1.png", "dormir_2.png"),
        "jugar": ("jugar_1.png", "jugar_2.png"),
    }
    frames = {
        key: [pygame.transform.smoothscale(pygame.image.load(str(asset_dir / file)).convert_alpha(), pet_size)
              for file in files]
        for key, files in frame_files.items()
    }

    needs = [0.2, 0.9, 0.2]
    slider_rects = [pygame.Rect(64, 219 + i * 77, 414, 16) for i in range(3)]
    minus_rects = [pygame.Rect(427, 180 + i * 77, 32, 32) for i in range(3)]
    plus_rects = [pygame.Rect(468, 180 + i * 77, 32, 32) for i in range(3)]
    random_rect = pygame.Rect(65, 432, 150, 39)
    reset_rect = pygame.Rect(227, 432, 150, 39)
    active_slider = None
    state = "calma"
    frame_index = 0
    last_frame = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    index = event.key - pygame.K_1
                    needs[index] = min(1.0, needs[index] + 0.1)
                elif event.key in (pygame.K_q, pygame.K_w, pygame.K_e):
                    index = {pygame.K_q: 0, pygame.K_w: 1, pygame.K_e: 2}[event.key]
                    needs[index] = max(0.0, needs[index] - 0.1)
                elif event.key == pygame.K_r:
                    needs = list(np.random.default_rng().random(3))
                elif event.key == pygame.K_n:
                    needs = [0.2, 0.9, 0.2]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if random_rect.collidepoint(event.pos):
                    needs = list(np.random.default_rng().random(3))
                elif reset_rect.collidepoint(event.pos):
                    needs = [0.2, 0.9, 0.2]
                else:
                    button_clicked = False
                    for i, rect in enumerate(minus_rects):
                        if rect.collidepoint(event.pos):
                            needs[i] = max(0.0, needs[i] - 0.1)
                            button_clicked = True
                            break
                    if not button_clicked:
                        for i, rect in enumerate(plus_rects):
                            if rect.collidepoint(event.pos):
                                needs[i] = min(1.0, needs[i] + 0.1)
                                button_clicked = True
                                break
                    if not button_clicked:
                        for i, rect in enumerate(slider_rects):
                            hitbox = rect.inflate(0, 12)
                            if hitbox.collidepoint(event.pos):
                                active_slider = i
                                needs[i] = float(np.clip((event.pos[0] - rect.left) / rect.width, 0, 1))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                active_slider = None
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, rect in enumerate(slider_rects):
                    if rect.inflate(0, 12).collidepoint(mouse_x, mouse_y):
                        needs[i] = float(np.clip(needs[i] + event.y * 0.05, 0, 1))
            elif event.type == pygame.MOUSEMOTION and active_slider is not None:
                rect = slider_rects[active_slider]
                needs[active_slider] = float(np.clip((event.pos[0] - rect.left) / rect.width, 0, 1))

        probs = probabilities_for(needs, model)
        action = ACTIONS[int(np.argmax(probs))]
        is_calm = needs[0] < 0.4 and needs[1] > 0.6 and needs[2] < 0.4
        new_state = "calma" if is_calm else action
        now = pygame.time.get_ticks()
        if new_state != state:
            state, frame_index, last_frame = new_state, 0, now
        elif now - last_frame > 520:
            frame_index = (frame_index + 1) % len(frames[state])
            last_frame = now

        screen.fill(BG)
        # A few soft, playful decorations frame the working area.
        pygame.draw.circle(screen, (255, 230, 239), (1110, 36), 18)
        pygame.draw.circle(screen, (228, 219, 255), (1080, 43), 8)
        draw_text(screen, title_font, "¡Hola! Soy Neurito", INK, (30, 22))
        draw_text(screen, body_font, "Mueve mis necesidades y descubre qué acción elige mi cerebro.", MUTED, (32, 62))
        draw_text(screen, small_font, "PEQUEÑO LABORATORIO DE IA", (161, 116, 196), (891, 39))

        left_card = pygame.Rect(28, 105, 510, 420)
        right_card = pygame.Rect(552, 105, 570, 420)
        controls_card = pygame.Rect(28, 545, 1094, 205)
        draw_card(screen, left_card, (255, 255, 255))
        draw_card(screen, right_card, (255, 255, 255))
        draw_card(screen, controls_card, LILAC)

        draw_text(screen, heading_font, "¿Qué necesita Neurito?", INK, (52, 124))
        draw_text(screen, small_font, "Arrastra una barra o usa los botones − y +", MUTED, (52, 154))
        mouse_pos = pygame.mouse.get_pos()
        for i, (label, color, value) in enumerate(zip(NEED_LABELS, NEED_COLORS, needs)):
            y = 183 + i * 77
            draw_text(screen, body_font, label, INK, (64, y))
            draw_text(screen, small_font, f"{round(value * 100)}%", MUTED, (369, y + 3))
            minus, plus = minus_rects[i], plus_rects[i]
            pygame.draw.circle(screen, (229, 220, 246), minus.center, 17)
            pygame.draw.circle(screen, (229, 220, 246), plus.center, 17)
            pygame.draw.circle(screen, (245, 240, 250), minus.center, 14)
            pygame.draw.circle(screen, (245, 240, 250), plus.center, 14)
            if minus.collidepoint(mouse_pos):
                pygame.draw.circle(screen, color, minus.center, 16, width=2)
            if plus.collidepoint(mouse_pos):
                pygame.draw.circle(screen, color, plus.center, 16, width=2)
            draw_text(screen, heading_font, "−", INK, minus.center, center=True)
            draw_text(screen, heading_font, "+", INK, plus.center, center=True)
            track = slider_rects[i]
            pygame.draw.rect(screen, (239, 234, 244), track, border_radius=9)
            fill = track.copy()
            fill.width = max(14, round(track.width * value))
            pygame.draw.rect(screen, color, fill, border_radius=9)
            knob_x = track.left + round(track.width * value)
            pygame.draw.circle(screen, PAPER, (knob_x, track.centery), 14)
            pygame.draw.circle(screen, color, (knob_x, track.centery), 14, width=4)

        for rect, label, fill in ((random_rect, "AZAR", GOLD), (reset_rect, "REINICIAR", PINK)):
            pygame.draw.rect(screen, fill, rect, border_radius=15)
            if rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, (208, 177, 226), rect, width=3, border_radius=15)
            draw_text(screen, small_font, label, INK, rect.center, center=True)

        draw_text(screen, heading_font, "¡Neurito está pensando!", INK, (578, 124))
        pet = frames[state][frame_index]
        screen.blit(pet, pet.get_rect(center=(720, 319)))
        label = ACTION_LABELS[state]
        action_color = ACTION_COLORS.get(action, (175, 141, 207))
        bubble = pygame.Rect(838, 174, 247, 56)
        pygame.draw.rect(screen, (255, 239, 247) if state == "calma" else action_color, bubble, border_radius=20)
        draw_text(screen, heading_font, label, INK, bubble.center, center=True)
        if state == "calma":
            draw_text(screen, small_font, "Todas mis necesidades están tranquilas.", MUTED, (840, 242))
        else:
            draw_text(screen, small_font, "Mi regla compara tres necesidades.", MUTED, (840, 242))
        draw_text(screen, small_font, "Lo que dicen mis salidas:", MUTED, (840, 278))
        for i, name in enumerate(ACTIONS):
            y = 310 + i * 43
            draw_text(screen, small_font, ACTION_LABELS[name].title(), INK, (840, y - 6))
            bar = pygame.Rect(925, y, 133, 13)
            pygame.draw.rect(screen, (240, 235, 244), bar, border_radius=8)
            color = ACTION_COLORS[name]
            bar.width = max(4, round(bar.width * float(probs[i])))
            pygame.draw.rect(screen, color, bar, border_radius=8)
            draw_text(screen, tiny_font, f"{probs[i]*100:.1f}%", MUTED, (1064, y - 4))
        draw_text(screen, heading_font, "Cómo jugar", INK, (53, 560))
        draw_text(screen, small_font, "Cambia las barras y mira cómo responde Neurito.", MUTED, (53, 590))
        help_items = (
            (pygame.Rect(52, 626, 245, 72), "ARRASTRAR", "Mueve el círculo de cada barra", (255, 255, 255)),
            (pygame.Rect(309, 626, 245, 72), "− / +", "Ajusta una necesidad poco a poco", (255, 255, 255)),
            (pygame.Rect(566, 626, 245, 72), "AZAR", "Prueba una combinación nueva", GOLD),
            (pygame.Rect(823, 626, 275, 72), "REINICIAR", "Vuelve al estado inicial", PINK),
        )
        for help_rect, help_title, help_body, help_color in help_items:
            pygame.draw.rect(screen, help_color, help_rect, border_radius=20)
            draw_text(screen, small_font, help_title, INK, (help_rect.centerx, help_rect.y + 21), center=True)
            draw_text(screen, tiny_font, help_body, MUTED, (help_rect.centerx, help_rect.y + 48), center=True)
        draw_text(screen, tiny_font, "1/2/3 = subir   ·   Q/W/E = bajar   ·   R = azar   ·   N = reiniciar   ·   rueda = ajustar bajo el puntero   ·   Esc = salir", MUTED, (575, 721), center=True)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
