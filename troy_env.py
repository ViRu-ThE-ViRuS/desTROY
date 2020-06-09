import pygame


# FPS = 1000
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

block_size = 10
clock = pygame.time.Clock()
game_display = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('desTROY')


class TroyEnv(object):
    def __init__(self):
        pygame.init()
        self.reset()

    def close(self):
        pygame.quit()
        quit()

    def reset(self):
        self.done = False
        self.rider1 = Rider(black, [window_width/2-50, window_height/2-50], up)
        self.rider2 = Rider(red, [window_width/2+50, window_height/2+50], down)

        return self.get_state()

    def step(self, action1, action2):
        assert not self.done

        self.done, reward = self.step_game(action1, action2)
        state = self.get_state()
        return state, reward, self.done, None

    def step_game(self, action1, action2):
        done = False
        reward = [0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        self.rider1.move(action1)
        self.rider2.move(action2)

        if self.rider1.out_of_bounds():
            done = True
            reward[1] = 100
        elif self.rider2.out_of_bounds():
            done = True
            reward[0] = 100

        self.rider1.advance()
        self.rider2.advance()
        game_display.fill(white)

        if self.rider1.check_self_collision():
            done = True
            reward[1] = 100
        elif self.rider2.check_self_collision():
            done = True
            reward[0] = 100

        if self.rider1.check_collision(self.rider2.components):
            done = True
            reward[1] = 100
        elif self.rider2.check_collision(self.rider1.components):
            done = True
            reward[0] = 100

        self.rider1.render()
        self.rider2.render()

        pygame.display.update()
        # clock.tick(FPS)

        return done, reward

    def get_state(self):
        return pygame.surfarray.array3d(game_display).swapaxes(0, 1)


class Rider:
    def __init__(self, color, position, direction):
        self.color = color
        self.lead = position
        self.direction = direction
        self.components = []

    def render(self):
        for component in self.components:
            pygame.draw.rect(game_display, self.color,
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

    def move(self, action):
        if action == 0:
            if self.direction == up:
                self.direction = left
            elif self.direction == down:
                self.direction = right
            elif self.direction == left:
                self.direction = down
            elif self.direction == right:
                self.direction = up
        elif action == 1:
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


if __name__ == '__main__':
    env = TroyEnv()

    for _ in range(10):
        done = False
        state = env.reset()
        while not done:
            state, reward, done, _ = env.step(0, 0)
            print(state, reward, done, _)

    env.close()
