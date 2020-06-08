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
        if np.argmax(action) == 0:
            self.aim.rotate(90)
        elif np.argmax(action) == 1:
            pass
        else:
            self.aim.rotate(-90)

    def update(self):
        self.xy.move(self.aim)
        head = self.xy.copy()
        self.body.add(head)
        return head

    def check(self, other):
        return other in self.body


class TroyEnv(object):
    # 1.x, 1.y, 1.aim.x, 1.aim.y, 2.x, 2.y, 2.aim.x, 2.aim.y
    initial_state = (-100, 0, 4, 0, 100, 0, -4, 0)
    observation_space = (8,)
    action_space = 3

    def __init__(self, dimen_x=200, dimen_y=200):
        self.dimen_x = dimen_x
        self.dimen_y = dimen_y

        self.reset()

    def reset(self):
        self.player1 = Player(p1xy_initial.copy(), p1aim_initial.copy())
        self.player2 = Player(p2xy_initial.copy(), p2aim_initial.copy())
        self._first = True

        self.done = 0
        self.state = self.initial_state
        return self.state

    def step(self, action1, action2):
        assert not self.done

        self.player1.move(action1)
        self.player2.move(action2)

        head1 = self.player1.update()
        head2 = self.player2.update()
        reward = 0

        if not self.inside(head1) or self.player2.check(head1):
            reward = -100
            self.done = True
        elif not self.inside(head2) or self.player1.check(head2):
            reward = +100
            self.done = True

        if done and not self._first:
            done()

        self.state = (self.player1.xy.x, self.player1.xy.y,
                      self.player1.aim.x, self.player1.aim.y,
                      self.player2.xy.x, self.player2.xy.y,
                      self.player2.aim.x, self.player2.aim.y)

        return self.state, reward, self.done, None

    def render(self):
        if self._first:
            self._first = False
            setup(self.dimen_x*2+20, self.dimen_y*2+20, 370, 370)
            hideturtle()
            tracer(False)

        square(self.player1.xy.x, self.player1.xy.y, 3, 'red')
        square(self.player2.xy.x, self.player2.xy.y, 3, 'blue')
        update()

    def inside(self, head):
        return -self.dimen_x < head.x < self.dimen-x and \
            -self.dimen_y < head.y < self.dimen_y
