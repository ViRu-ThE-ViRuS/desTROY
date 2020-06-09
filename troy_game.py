import pygame

pygame.init()

white = pygame.Color(255, 255, 255)
black = pygame.Color(0, 0, 0)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
window_height = 600
window_width = 600

up = (0, -1)
down = (0, 1)
left = (-1, 0)
right = (1, 0)

gameDisplay = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('desTROY')

clock = pygame.time.Clock()
FPS = 15
block_size = 10


# class Rider:
# def __init__(self, color, position):
# self.color = color
# self.components = position

# def render(self):
# for component in self.components:
# pygame.draw.rect(gameDisplay, black, [component[0], component[1],
# block_size, block_size])


def rider(block_size, components):
    for component in components:
        pygame.draw.rect(gameDisplay, black, [component[0], component[1],
                                              block_size, block_size])


def gameLoop():
    gameExit = False

    lead_x = window_width / 2
    lead_y = window_height / 2

    components = []
    direction = up

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True

            if event.type == pygame.KEYDOWN:
                left_command = event.key == pygame.K_j
                right_command = event.key == pygame.K_l

                if left_command:
                    if direction == up:
                        direction = left
                    elif direction == down:
                        direction = right
                    elif direction == left:
                        direction = down
                    elif direction == right:
                        direction = up
                elif right_command:
                    if direction == up:
                        direction = right
                    elif direction == down:
                        direction = left
                    elif direction == left:
                        direction = up
                    elif direction == right:
                        direction = down

        if lead_x >= window_width or lead_x < 0 or \
                lead_y >= window_height or lead_y < 0:
            gameExit = True

        lead_x += direction[0] * block_size
        lead_y += direction[1] * block_size

        gameDisplay.fill(white)
        components.append((lead_x, lead_y))

        for each_segment in components[:-1]:
            if each_segment == (lead_x, lead_y):
                gameExit = True

        rider(block_size, components)
        pygame.display.update()
        clock.tick(FPS)

        # pixData = pygame.surfarray.array3d(gameDisplay).swapaxes(0, 1).shape

    pygame.quit()
    quit()


gameLoop()
