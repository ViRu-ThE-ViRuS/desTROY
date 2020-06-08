import time
import numpy as np

from turtle import *
from freegames import vector, square

p1xy_initial = vector(-100, 0)
p1aim_initial = vector(4, 0)
p2xy_initial = vector(100, 0)
p2aim_initial = vector(-4, 0)


class Player(object):
    def __init__(self, xy, aim):
        self.xy = xy
        self.aim = aim
        self.body = set()

    def move(self, action):
        if action == 0:
            self.aim.rotate(90)
        elif action == 1:
            pass
        else:
            self.aim.rotate(-90)

    def update(self):
        self.xy.move(self.aim)
        head = self.xy.copy()
        return head

    def post_update(self, head):
        self.body.add(head)

    def check(self, other):
        return other in self.body


class TroyEnv(object):
    # 1.x, 1.y, 1.aim.x, 1.aim.y, 2.x, 2.y, 2.aim.x, 2.aim.y
    initial_state = (-100, 0, 4, 0, 100, 0, -4, 0)
    observation_space = (8,)
    action_space = 3

    def __init__(self, dimen_x=200, dimen_y=200, manual=False):
        self.dimen_x = dimen_x
        self.dimen_y = dimen_y
        self.manual = True

        self.reset()

    def reset(self):
        self.player1 = Player(p1xy_initial.copy(), p1aim_initial.copy())
        self.player2 = Player(p2xy_initial.copy(), p2aim_initial.copy())

        self.done = 0
        self.state = self.initial_state
        return self.state

    def _default_step(self):
        head1 = self.player1.update()
        head2 = self.player2.update()

        if not self.inside(head1) or self.player2.check(head1) or \
                self.player1.check(head1):
            reward = -100
            self.done = True
        elif not self.inside(head2) or self.player1.check(head2) or \
                self.player2.check(head2):
            reward = +100
            self.done = True

        self.player1.post_update(head1)
        self.player2.post_update(head2)

        if self.done:
            self.state = (self.player1.xy.x, self.player1.xy.y,
                          self.player1.aim.x, self.player1.aim.y,
                          self.player2.xy.x, self.player2.xy.y,
                          self.player2.aim.x, self.player2.aim.y)
            return self.state, reward, self.done, None

        return None

    def step(self, action1, action2):
        assert not self.done

        returns = self._default_step()
        if returns:
            return returns

        self.player1.move(action1)
        self.player2.move(action2)

        head1 = self.player1.update()
        head2 = self.player2.update()
        reward = 0

        if not self.inside(head1) or self.player2.check(head1) or \
                self.player1.check(head1):
            reward = -100
            self.done = True
        elif not self.inside(head2) or self.player1.check(head2) or \
                self.player2.check(head2):
            reward = +100
            self.done = True

        self.player1.post_update(head1)
        self.player2.post_update(head2)

        self.state = (self.player1.xy.x, self.player1.xy.y,
                      self.player1.aim.x, self.player1.aim.y,
                      self.player2.xy.x, self.player2.xy.y,
                      self.player2.aim.x, self.player2.aim.y)

        return self.state, reward, self.done, None

    def _draw(self):
        head1 = self.player1.update()
        head2 = self.player2.update()

        if not self.inside(head1) or self.player2.check(head1) or \
                self.player1.check(head1):
            self.done = True
        elif not self.inside(head2) or self.player1.check(head2) or \
                self.player2.check(head2):
            self.done = True

        self.player1.post_update(head1)
        self.player2.post_update(head2)

        square(self.player1.xy.x, self.player1.xy.y, 3, 'red')
        square(self.player2.xy.x, self.player2.xy.y, 3, 'blue')

        update()

        if self.done:
            done()
        else:
            ontimer(self._draw, 50)

    def render(self):
        assert self.manual

        setup(420, 420, 370, 0)
        hideturtle()
        tracer(False)
        listen()
        onkey(lambda: self.player2.aim.rotate(90), 'j')
        onkey(lambda: self.player2.aim.rotate(-90), 'l')
        self._draw()
        done()

    def inside(self, head):
        return -self.dimen_x < head.x < self.dimen_x and \
            -self.dimen_y < head.y < self.dimen_y
