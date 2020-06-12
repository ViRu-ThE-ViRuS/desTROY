import pygame
from PIL import Image
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

    def reset(self, single_player=False):
        self.done = False
        self.single_player = single_player
        self.rider1 = Rider(green, [window_width/2-50, window_height/2-50], up)
        if not self.single_player:
            self.rider2 = Rider(
                red, [window_width/2+50, window_height/2+50], down)

        game_display.fill(white)
        self.draw_border(black)

        return self.get_state()

    def step(self, action1, action2, render=False):
        assert not self.done

        self.done, reward = self.step_game(action1, action2, render)
        state = self.get_state()
        return state, reward, self.done, None

    def step_game(self, action1, action2, render):
        done = False
        reward = [0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        self.rider1.move(action1)
        if not self.single_player:
            self.rider2.move(action2)

        if self.rider1.out_of_bounds():
            done = True
            reward[0] = -100
        if not self.single_player and self.rider2.out_of_bounds():
            done = True
            reward[1] = -100

        self.rider1.advance()
        if not self.single_player:
            self.rider2.advance()

        game_display.fill(white)
        self.draw_border(black)

        if self.rider1.check_self_collision():
            done = True
            reward[0] = -100

        if not self.single_player and self.rider2.check_self_collision():
            done = True
            reward[1] = -100

        if self.rider1.check_collision(self.rider2.components):
            done = True
            reward[0] = -100
        if not self.single_player and self.rider2.check_collision(self.rider1.components):
            done = True
            reward[1] = -100

        self.rider1.render()
        if not self.single_player:
            self.rider2.render()

        if render:
            pygame.display.update()
            # clock.tick(FPS)

        return done, np.array(reward)

    def get_state(self):
        dimen1 = [self.rider1.lead[0] - 50,
                  self.rider1.lead[1] - 100,
                  self.rider1.lead[0] + 50 + block_size,
                  self.rider1.lead[1] + block_size]

        if not self.single_player:
            dimen2 = [self.rider2.lead[0] - 50,
                      self.rider2.lead[1] - 100,
                      self.rider2.lead[0] + 50 + block_size,
                      self.rider2.lead[1] + block_size]

        image = np.array(pygame.surfarray.array3d(game_display).swapaxes(0, 1))
        image = Image.fromarray(image).convert('L')
        cropped_images = [image.crop(dimen1), image.crop(dimen2) if
                          not self.single_player else None]
        np_images = list(map(self.image_to_np, cropped_images))
        return list(map(TroyEnv.image_dimen_swap, np_images))

    def draw_border(self, color):
        pygame.draw.rect(game_display,
                         color, [0, 0, window_width, block_size])
        pygame.draw.rect(game_display,
                         color, [0, 0, block_size, window_height])
        pygame.draw.rect(game_display,
                         color, [0, window_height-block_size, window_width, block_size])
        pygame.draw.rect(game_display,
                         color, [window_width-block_size, 0, block_size, window_height])

    def image_to_np(self, image):
        if image is not None:
            return (np.array(image.getdata())/255.0).reshape(self.observation_space[::-1])
        return None

    def image_dimen_swap(image):
        if image is not None:
            return image.swapaxes(0, 2)
        return None


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
        if self.lead[0] >= window_width-2*block_size or self.lead[0] <= 2*block_size or \
                self.lead[1] >= window_height-2*block_size or self.lead[1] <= 2*block_size:
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
