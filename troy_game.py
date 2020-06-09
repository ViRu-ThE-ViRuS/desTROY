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
FPS = 10
block_size = 10


class Rider:
    def __init__(self, color, position, direction):
        self.color = color
        self.lead = position
        self.direction = direction
        self.components = []

    def render(self):
        for component in self.components:
            pygame.draw.rect(gameDisplay, self.color,
                             [component[0], component[1],
                              block_size, block_size])

    def handle_input(self, event, leftkey, rightkey):
        if event.type == pygame.KEYDOWN:
            left_command = event.key == leftkey
            right_command = event.key == rightkey

            if left_command:
                if self.direction == up:
                    self.direction = left
                elif self.direction == down:
                    self.direction = right
                elif self.direction == left:
                    self.direction = down
                elif self.direction == right:
                    self.direction = up
            elif right_command:
                if self.direction == up:
                    self.direction = right
                elif self.direction == down:
                    self.direction = left
                elif self.direction == left:
                    self.direction = up
                elif self.direction == right:
                    self.direction = down

    def out_of_bounds(self):
        if self.lead[0] >= window_width or self.lead[0] < 0 or \
                self.lead[1] >= window_height or self.lead[1] < 0:
            return True

        return False

    def advance(self):
        self.lead[0] += self.direction[0] * block_size
        self.lead[1] += self.direction[1] * block_size
        self.components.append((self.lead[0], self.lead[1]))

    def check_self_collision(self):
        for each_segment in self.components[:-1]:
            if each_segment == (self.lead[0], self.lead[1]):
                return True

        return False

    def check_collision(self, other_components):
        for each_segment in other_components:
            if each_segment == (self.lead[0], self.lead[1]):
                return True
        return False


def step_game(rider1, rider2):
    gameExit = False
    reward = [0, 0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameExit = True

        rider1.handle_input(event, pygame.K_j, pygame.K_l)
        rider2.handle_input(event, pygame.K_a, pygame.K_d)

    if rider1.out_of_bounds():
        gameExit = True
        reward[1] = 100
    elif rider2.out_of_bounds():
        gameExit = True
        reward[0] = 100

    rider1.advance()
    rider2.advance()
    gameDisplay.fill(white)

    if rider1.check_self_collision():
        gameExit = True
        reward[1] = 100
    elif rider2.check_self_collision():
        gameExit = True
        reward[0] = 100

    if rider1.check_collision(rider2.components):
        gameExit = True
        reward[1] = 100
    elif rider2.check_collision(rider1.components):
        gameExit = True
        reward[0] = 100

    rider1.render()
    rider2.render()

    pygame.display.update()
    clock.tick(FPS)

    # pixData = pygame.surfarray.array3d(gameDisplay).swapaxes(0, 1).shape
    return gameExit, reward


gameExit = False
rider1 = Rider(black, [window_width/2-50, window_height/2-50], up)
rider2 = Rider(red, [window_width/2+50, window_height/2+50], down)

while not gameExit:
    gameExit, reward = step_game(rider1, rider2)
    print(reward)
    if gameExit:
        print('exit')

pygame.quit()
quit()
