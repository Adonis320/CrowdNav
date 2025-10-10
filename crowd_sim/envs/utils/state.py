class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])


class JointState(object):
    def __init__(self, self_state, human_states, obstacles=None):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = [
                (o if isinstance(o, ObstacleRect) else ObstacleRect(*o))
                for o in obstacles
            ]

    # Convenience: obstacles in robot-centric coordinates (centered at self_state.px,py)
    def obstacles_relative(self):
        px, py = self.self_state.px, self.self_state.py
        # Each item: (cx_rel, cy_rel, hx, hy)
        return [o.relative_to(px, py) for o in self.obstacles]

    # Optional: flat vector of first K obstacles (robot-centric), padded with zeros
    def obstacles_vector(self, k_max=0):
        """
        Returns a flat tuple: (cx1, cy1, hx1, hy1, cx2, cy2, hx2, hy2, ...),
        length = 4 * k_max. If k_max==0, return all (variable length).
        """
        rel = self.obstacles_relative()
        if k_max <= 0 or k_max >= len(rel):
            # variable-length (use with care)
            return tuple(v for item in rel for v in item)
        out = []
        for i in range(k_max):
            if i < len(rel):
                out.extend(rel[i])
            else:
                out.extend((0.0, 0.0, 0.0, 0.0))
        return tuple(out)

class ObstacleRect(object):
    """
    Axis-aligned rectangular obstacle (xmin, xmax, ymin, ymax).
    Lightweight container with convenience accessors.
    """
    __slots__ = ("xmin", "xmax", "ymin", "ymax")
    def __init__(self, xmin, xmax, ymin, ymax):
        assert xmin < xmax and ymin < ymax
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

    @property
    def as_tuple(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def center(self):
        cx = 0.5 * (self.xmin + self.xmax)
        cy = 0.5 * (self.ymin + self.ymax)
        return (cx, cy)

    @property
    def half_extents(self):
        hx = 0.5 * (self.xmax - self.xmin)
        hy = 0.5 * (self.ymax - self.ymin)
        return (hx, hy)

    def relative_to(self, px, py):
        """
        Return (cx_rel, cy_rel, hx, hy) where (cx_rel, cy_rel) is obstacle center
        in coordinates centered at (px, py); hx, hy are half-sizes (unchanged).
        """
        (cx, cy) = self.center
        (hx, hy) = self.half_extents
        return (cx - px, cy - py, hx, hy)

    def __str__(self):
        return f"{self.xmin} {self.xmax} {self.ymin} {self.ymax}"

    # Optional: support tuple-like concatenation if you use __add__ patterns
    def __add__(self, other):
        # so you can do: other + (xmin, xmax, ymin, ymax)
        return other + self.as_tuple