import pygame
from PIL import Image, ImageOps
import numpy as np

FPS = 100
white = pygame.Color(255, 255, 255)
black = pygame.Color(0, 0, 0)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
window_height = 200
window_width = 200

up = (0, -1)
down = (0, 1)
left = (-1, 0)
right = (1, 0)

block_size = 10
clock = pygame.time.Clock()
game_display = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('desTROY')


class TroyEnv(object):
    action_space = 3
    observation_space = (1, 110, 110)

    def __init__(self):
        pygame.init()
        self.reset()

    def close(self):
        pygame.quit()
        quit()

    def reset(self):
        random_x = np.random.randint(window_height/2-50, window_height/2+50)
        random_y = np.random.randint(window_width/2-50, window_width/2+50)

        self.done = False
        self.rider = Rider(green, [random_x, random_y], up)

        game_display.fill(white)
        self.draw_border(black)
        self.rider.advance()
        self.rider.render()
        return self.get_state()

    def step(self, action, render=False):
        assert not self.done

        self.done, reward = self.step_game(action, render)
        state = self.get_state()
        return state, reward, self.done, None

    def step_game(self, action, render):
        done = False
        reward = [0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        self.rider.move(action)
        self.rider.advance()

        if self.rider.out_of_bounds() or \
                self.rider.check_self_collision():
            done = True
            reward[0] = -100

        game_display.fill(white)
        self.draw_border(black)

        self.rider.render()

        if render:
            pygame.display.update()
            # clock.tick(FPS)

        return done, np.array(reward)

    def get_state(self):
        dimen1 = self.get_agent_vision_dimens()

        image = np.array(pygame.surfarray.array3d(game_display).swapaxes(0, 1))
        image = Image.fromarray(image)
        image = ImageOps.grayscale(image)

        # left, upper, right, lower
        # cropped_images = [image.crop(dimen1), image.crop(dimen2)]
        cropped_images = [image.crop(dimen1)]

        np_images = list(map(self.image_to_np, cropped_images))
        return list(map(TroyEnv.image_dimen_swap, np_images))

    def draw_border(self, color):
        pygame.draw.rect(game_display, color, [0, 0, window_width, block_size])
        pygame.draw.rect(game_display, color, [0, 0, block_size, window_height])
        pygame.draw.rect(game_display, color, [0, window_height-block_size, window_width, block_size])
        pygame.draw.rect(game_display, color, [window_width-block_size, 0, block_size, window_height])

    def image_to_np(self, image):
        if image is not None:
            return (np.array(image.getdata())/255.0).reshape(self.observation_space[::-1])
        return None

    def image_dimen_swap(image):
        if image is not None:
            return image.swapaxes(0, 2)
        return None

    def get_agent_vision_dimens(self):
        # TODO: localize

        dimen1 = [
            self.rider.lead[0] - 50,
            self.rider.lead[1] - 50,
            self.rider.lead[0] + 50 + block_size,
            self.rider.lead[1] + 50 + block_size
        ]

        return dimen1


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
            pass  # no op
        elif action == 2:
            if self.direction == up:
                self.direction = right
            elif self.direction == down:
                self.direction = left
            elif self.direction == left:
                self.direction = up
            elif self.direction == right:
                self.direction = down

    def out_of_bounds(self):
        return self.lead[0] >= window_width-2*block_size or self.lead[0] <= block_size or \
            self.lead[1] >= window_height-2*block_size or self.lead[1] <= block_size

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
