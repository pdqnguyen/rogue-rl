import time
import yaml
import numpy as np

import gym
from gym import spaces

import pyglet
from pyglet import shapes
from pyglet import gl

from .rendering import Transform


# Game parameters
STEP = 1
FPS = 60.0
MAX_ATTEMPTS = 100  # max attempts when initializing world and objects
MOVES = [
    np.array([0, 0]),
    np.array([-STEP, 0]),
    np.array([STEP, 0]),
    np.array([0, STEP]),
    np.array([0, -STEP])
]


def distance(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def distance_to_line(point, line):
    x0, y0 = point
    x1, y1, x2, y2 = line
    num = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    den = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return num / den


def rect2box(rect):
    return rect.x, rect.y, rect.width, rect.height


def box_overlap(box, geom, min_ovl=0):
    x1, y1, w1, h1 = box
    assert len(geom) == 2 or len(geom) == 4
    if len(geom) == 2:
        # Box and single point
        x2, y2 = geom
        if x1 + min_ovl <= x2 <= x1 + w1 - min_ovl and y1 + min_ovl <= y2 <= y1 + h1 - min_ovl:
            return True
    else:
        # Box and box
        x2, y2, w2, h2 = geom
        corners = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]
        for c in corners:
            if x1 + min_ovl <= c[0] <= x1 + w1 - min_ovl and y1 + min_ovl <= c[1] <= y1 + h1 - min_ovl:
                return True
    return False


def box_kissing(box_1, box_2):
    x1, y1, w1, h1 = box_1
    x2, y2, w2, h2 = box_2
    if min(abs((x1 + w1) - x2), abs(x1 - (x2 + w2))) < 2:
        return True
    elif min(abs((y1 + h1) - y2), abs(y1 - (y2 + h2))) < 2:
        return True
    else:
        return False


def world_area(rectangles):
    def compare(r1, r2):
        intersection_x, intersection_y = 0, 0
        if not (r2.x > r1.x + r1.width or r2.x + r2.width < r1.x):
            intersection_x = min(r1.x + r1.width, r2.x + r2.width) - max(r1.x, r2.x)
        if not (r2.y > r1.y + r1.width or r2.y + r2.width < r1.y):
            intersection_y = min(r1.y + r1.width, r2.y + r2.width) - max(r1.y, r2.y)
        return intersection_x * intersection_y
    n = len(rectangles)
    area = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                area += compare(rectangles[i], rectangles[j])
    return area


class World:
    def __init__(self, batch=None, **kwargs):
        self.width, self.height = kwargs.pop("size")
        self.params = kwargs
        self.params["min_area"] = self.params["min_area_factor"] * self.width * self.height
        self.rectangles = []
        self._generate(batch=batch)
        self.area = world_area(self.rectangles)

    def _random_box(self, max_attempts=MAX_ATTEMPTS):
        x_min = -self.width // 2
        x_max = self.width // 2
        y_min = -self.height // 2
        y_max = self.height // 2
        min_segment_area = self.params["min_segment_area_factor"] * self.params["min_area"]
        max_segment_area = self.params["max_segment_area_factor"] * self.params["min_area"]
        min_segment_dim = int(self.params["min_segment_dim_factor"] * min(self.width, self.height))
        max_segment_dim = int(self.params["max_segment_dim_factor"] * min(self.width, self.height))
        area = np.random.randint(min_segment_area, max_segment_area)
        for _ in range(max_attempts):
            if np.random.uniform() < 0.5:
                w = np.random.randint(min_segment_dim, max_segment_dim)
                h = area // w
            else:
                h = np.random.randint(min_segment_dim, max_segment_dim)
                w = area // h
            if w < self.width and h < self.height:
                break
        x = np.random.randint(x_min, x_max - w)
        y = np.random.randint(y_min, y_max - h)
        return x, y, w, h

    def _generate(self, batch=None, max_attempts=MAX_ATTEMPTS):
        min_ovl = self.params["min_ovl"]
        color = self.params["color"]
        while True:
            initial_box = self._random_box()
            if not box_overlap(initial_box, (0, 0), min_ovl=min_ovl):
                continue
            initial_rect = shapes.Rectangle(*initial_box, color=color, batch=batch)
            self.rectangles.append(initial_rect)
            for _ in range(max_attempts):
                for _ in range(max_attempts):
                    new_box = self._random_box()
                    good = False
                    for rect in self.rectangles:
                        box = rect2box(rect)
                        if box_overlap(box, new_box, min_ovl=min_ovl):
                            good = True
                        if box_kissing(box, new_box):
                            good = False
                            break
                    if good:
                        new_rect = shapes.Rectangle(*new_box, color=color, batch=batch)
                        self.rectangles.append(new_rect)
                if world_area(self.rectangles) >= self.params["min_area"]:
                    break
            if len(self.rectangles) > 1:
                break
        return


class GameObject:
    def __init__(self):
        pass

    def collides_with(self, other):
        if isinstance(other, shapes.Circle):
            d_min = self.radius + other.radius
            d = distance(self.position, other.position)
            return d <= d_min
        elif isinstance(other, World):
            collides = False
            for rect in other.rectangles:
                collides = box_overlap(rect2box(rect), self.position, min_ovl=0)
                if collides:
                    break
            return collides
        else:
            raise TypeError(f"invalid type for `other`: {type(other)}")


class Player(shapes.Circle, GameObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = dict(left=False, right=False, up=False, down=False)
        self.previous = self.position
        self.tiles_visited = []

    def step(self, move):
        self.previous = self.position
        self.x += move[0]
        self.y += move[1]

    def undo(self):
        self.position = self.previous

    def handle_collision(self, other):
        if isinstance(other, Goal):
            return 'goal'
        elif isinstance(other, Enemy):
            return 'enemy'


class Goal(shapes.Circle, GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Enemy(shapes.Circle, GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RogueLike(gym.Env):
    def __init__(self, config=None):
        self._get_config(config)
        self.window = None
        self.batch = pyglet.graphics.Batch()
        self.world = None
        self.player = None
        self.goal = None
        self.enemies = []
        self.action_space = spaces.Discrete(5)
        state_w, state_h = self.config["state"]["size"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(state_h, state_w, 4), dtype=np.uint8
        )

    def _get_config(self, config):
        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.config = cfg["game"]

    @property
    def game_objects(self):
        obs = []
        if self.world is not None:
            obs.append(self.world)
        if self.player is not None:
            obs.append(self.player)
        if self.goal is not None:
            obs.append(self.goal)
        obs += self.enemies
        return obs

    def reset(self):
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        while True:
            self._generate_world()
            success = self._generate_game_objects()
            if success:
                break
            print("failed to place game objects; recreating game world")
        return self.step(None)[0]

    def _generate_world(self):
        self.batch = pyglet.graphics.Batch()
        kwargs = self.config["world"]
        kwargs["min_ovl"] = 2 * self.config["character"]["size"]
        self.world = World(batch=self.batch, **kwargs)

    def _random_valid_point(self, max_attempts=100*MAX_ATTEMPTS, size=0):
        size = int(size)
        for _ in range(max_attempts):
            x = np.random.randint(-self.world.width // 2 + size, self.world.width // 2 - size)
            y = np.random.randint(-self.world.height // 2 + size, self.world.height // 2 - size)
            boxes = [rect2box(r) for r in self.world.rectangles]
            if any(box_overlap(box, (x, y), min_ovl=size) for box in boxes):
                return x, y

    def _generate_game_objects(self):
        char_cfg = self.config["character"]
        char_size = char_cfg["size"]
        num_enemies = char_cfg["num_enemies"]
        enemy_color = char_cfg["enemy_color"]
        goal_cfg = self.config["goal"]
        goal_size = goal_cfg["size"]
        goal_color = goal_cfg["color"]
        self.player = Player(0, 0, char_size, batch=self.batch)
        import itertools
        for _ in range(MAX_ATTEMPTS):
            self.goal = None
            self.enemies = []
            goal_pos = self._random_valid_point(size=goal_size)
            if goal_pos is None:
                continue
            self.goal = Goal(goal_pos[0], goal_pos[1], goal_size, color=goal_color, batch=self.batch)
            self.enemies = []
            while len(self.enemies) < num_enemies:
                enemy_pos = self._random_valid_point(size=char_size)
                if enemy_pos is None:
                    continue
                enemy = Enemy(enemy_pos[0], enemy_pos[1], char_size, color=enemy_color, batch=self.batch)
                self.enemies.append(enemy)
            # Check for collisions between each circle object and each other game object
            combos = itertools.combinations(range(len(self.game_objects)), r=2)
            collisions = False
            for i, j in combos:
                ob1 = self.game_objects[i]
                ob2 = self.game_objects[j]
                if isinstance(ob1, GameObject):
                    if ob1.collides_with(ob2):
                        collisions = True
                        break
            if not collisions:
                break
        return self.goal is not None and len(self.enemies) == num_enemies

    def _get_state_coords(self, pos):
        state_w, state_h = self.config["state"]["size"]
        x_offset = state_w // 2 - self.player.x
        y_offset = state_h // 2 - self.player.y
        return pos[1] + y_offset, pos[0] + x_offset

    def step(self, action):
        self.t += 1.0 / FPS
        collisions = []
        if action is not None:
            move = MOVES[action]
            self.player.step(move)
            for ob in self.game_objects:
                if ob != self.player:
                    if self.player.collides_with(ob):
                        col = self.player.handle_collision(ob)
                        if col is not None:
                            collisions.append(col)
                    elif ob == self.world:
                        self.player.undo()
        self.state = self.render("state_pixels")
        step_reward = 0
        done = False
        reward_dict = self.config["reward"]
        if action is not None:
            i, j = self._get_state_coords(self.player.position)
            if self.state[i, j, 3]:
                self.reward += reward_dict["explore"]
            if (
                    self.player.previous != self.player.position and
                    self.player.previous not in self.player.tiles_visited
            ):
                self.player.tiles_visited.append(self.player.previous)
            if len(self.player.tiles_visited) == self.world.area:
                done = True
            self.reward -= reward_dict["step"]
            for col in collisions:
                if col == "goal":
                    done = True
                    self.reward = reward_dict["win"]
                    break
                elif col == "enemy":
                    done = True
                    self.reward = reward_dict["lose"]
                    break
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
        return self.state, step_reward, done, {}

    def render(self, mode="human", save_video=False):
        assert mode in ["human", "state_pixels", "rgb_array"]
        screen_w, screen_h = self.config["screen"]["size"]
        state_w, state_h = self.config["state"]["size"]
        zoom = self.config["zoom"]
        window_w = state_w * zoom
        window_h = state_h * zoom
        window_x = (screen_w - window_w) // 2
        window_y = (screen_h - window_h) // 2
        if self.window is None:
            self.window = pyglet.window.Window(window_w, window_h, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
            self.window.set_location(window_x, window_y)
            self.transform = Transform()
        pyglet.clock.tick()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            window_w / 2 - zoom * self.player.x,
            window_h / 2 - zoom * self.player.y,
        )

        win = self.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W, VP_H = self.config["video"]["size"]
        elif mode == "state_pixels":
            VP_W = state_w
            VP_H = state_h
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * window_w)
            VP_H = int(pixel_scale * window_h)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.batch.draw()
        t.disable()

        if mode == "human":
            time.sleep(1.0 / FPS)
            win.flip()
            return self.window is not None

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, :]
        visited = []
        for pos in self.player.tiles_visited:
            i, j = self._get_state_coords(pos)
            if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
                visited.append((pos, i, j))
                arr[i, j, 3] = 1
        return arr

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None


if __name__ == "__main__":
    from pyglet.window import key

    action = 0

    def key_press(k, mod):
        global action, restart, env
        if k == key.BACKSPACE:
            restart = True
        if k == key.ESCAPE:
            env.close()
        if k == key.A:
            action = 1
        if k == key.D:
            action = 2
        if k == key.W:
            action = 3
        if k == key.S:
            action = 4

    def key_release(k, mod):
        global action
        if k == key.A and action != 0:
            action = 0
        if k == key.D and action != 0:
            action = 0
        if k == key.W and action != 0:
            action = 0
        if k == key.S and action != 0:
            action = 0

    env = RogueLike()
    env.render()
    env.window.on_key_press = key_press
    env.window.on_key_release = key_release
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(action)
            total_reward += r
            steps += 1
            isopen = env.render()
            if done or restart or not isopen:
                break
    env.close()
