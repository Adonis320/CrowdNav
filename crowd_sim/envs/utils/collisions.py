# crowd_sim/envs/utils/collisions.py

import math

# --- Bounds: center must stay within bounds shrunk by r -----------------------
def project_within_bounds(x, y, r, bounds):
    xmin, xmax, ymin, ymax = bounds
    xmin2, xmax2 = xmin + r, xmax - r
    ymin2, ymax2 = ymin + r, ymax - r
    return (min(max(x, xmin2), xmax2),
            min(max(y, ymin2), ymax2))

# --- Obstacles: rectangles are AABBs (xmin,xmax,ymin,ymax) --------------------

def _inflate_rect(rect, r):
    xmin, xmax, ymin, ymax = rect
    return (xmin - r, xmax + r, ymin - r, ymax + r)

def inside_rect(x, y, rect):
    xmin, xmax, ymin, ymax = rect
    return (x > xmin) and (x < xmax) and (y > ymin) and (y < ymax)

def project_outside_rect(x, y, rect, r):
    """
    If (x,y) center lies *inside* rect inflated by r, push it to the nearest boundary,
    leaving the center just outside that inflated rect.
    """
    ir = _inflate_rect(rect, r)
    if not inside_rect(x, y, ir):
        return x, y
    xmin, xmax, ymin, ymax = ir
    # distances to each side
    dx_left   = abs(x - xmin)
    dx_right  = abs(xmax - x)
    dy_bottom = abs(y - ymin)
    dy_top    = abs(ymax - y)
    m = min(dx_left, dx_right, dy_bottom, dy_top)
    if m == dx_left:
        x = xmin
    elif m == dx_right:
        x = xmax
    elif m == dy_bottom:
        y = ymin
    else:
        y = ymax
    # retreat a hair outside to avoid sticking
    eps = 1e-6
    if m == dx_left:   x -= eps
    if m == dx_right:  x += eps
    if m == dy_bottom: y -= eps
    if m == dy_top:    y += eps
    return x, y

def project_outside_obstacles(x, y, r, obstacles, max_iters=3):
    """
    Repeatedly push the center out of any rectangles (inflated by r).
    Returns (x,y, bumped_any).
    """
    bumped_any = False
    for _ in range(max_iters):
        moved = False
        for rect in obstacles or []:
            nx, ny = project_outside_rect(x, y, rect, r)
            if (abs(nx - x) > 1e-12) or (abs(ny - y) > 1e-12):
                x, y = nx, ny
                moved = True
                bumped_any = True
        if not moved:
            break
    return x, y, bumped_any

# --- Swept-circle (segment) vs AABB (inflated by r) ---------------------------

def _segment_aabb_first_hit(x0, y0, x1, y1, rect):
    """
    Liang–Barsky / slab method for segment vs AABB.
    Returns (hit:bool, t_entry:float in [0,1]) for first intersection.
    """
    xmin, xmax, ymin, ymax = rect
    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0

    # X slabs
    if abs(dx) < 1e-12:
        if x0 < xmin or x0 > xmax:
            return False, 1.0
    else:
        tx1 = (xmin - x0) / dx
        tx2 = (xmax - x0) / dx
        tmin = min(tx1, tx2)
        tmax = max(tx1, tx2)
        t0 = max(t0, tmin)
        t1 = min(t1, tmax)
        if t1 < t0:
            return False, 1.0

    # Y slabs
    if abs(dy) < 1e-12:
        if y0 < ymin or y0 > ymax:
            return False, 1.0
    else:
        ty1 = (ymin - y0) / dy
        ty2 = (ymax - y0) / dy
        tmin = min(ty1, ty2)
        tmax = max(ty1, ty2)
        t0 = max(t0, tmin)
        t1 = min(t1, tmax)
        if t1 < t0:
            return False, 1.0

    # If segment starts inside the AABB, treat as hit at t=0
    return True, max(0.0, t0)

def move_and_project(x0, y0, x1, y1, r, bounds, obstacles):
    """
    Move center along segment (x0,y0)->(x1,y1), respecting:
      - bounds shrunk by r
      - rectangular obstacles inflated by r
    Returns (fx, fy, hit_wall, hit_obst).
    """
    # 1) clamp desired end to shrunk bounds
    x1, y1 = project_within_bounds(x1, y1, r, bounds)
    hit_wall = (abs(x1 - (min(max(x1, bounds[0]+r), bounds[1]-r))) > 0) or \
               (abs(y1 - (min(max(y1, bounds[2]+r), bounds[3]-r))) > 0)

    # 2) earliest obstacle hit along the path
    t_hit = 1.0
    hit_any = False
    for rect in obstacles or []:
        ir = _inflate_rect(rect, r)
        hit, t_entry = _segment_aabb_first_hit(x0, y0, x1, y1, ir)
        if hit and 0.0 <= t_entry <= t_hit:
            t_hit = t_entry
            hit_any = True

    eps = 1e-6
    if hit_any:
        # stop just before contact
        t_stop = max(0.0, min(t_hit - (eps / max(1e-12, math.hypot(x1-x0, y1-y0))), 1.0))
        fx = x0 + (x1 - x0) * t_stop
        fy = y0 + (y1 - y0) * t_stop
        # also push out in case we started inside due to numerical issues
        fx, fy, _ = project_outside_obstacles(fx, fy, r, obstacles)
        return fx, fy, hit_wall, True

    # no obstacle hit
    return x1, y1, hit_wall, False

# --- Clearance for reward/danger (capsule vs rect) ----------------------------

def capsule_to_rect_clearance(x0, y0, x1, y1, rect, r):
    """
    Conservative clearance between a swept circle (segment + radius) and an axis-aligned rect.
    Equivalent to segment vs *inflated* rect distance.
    Returns signed distance (>0 safe, ~0 touching, <0 penetrates).
    """
    # Inflate rect by r and compute distance from segment center-line to that AABB.
    xmin, xmax, ymin, ymax = _inflate_rect(rect, r)
    # Segment AABB distance
    # clamp segment endpoints into rect’s space
    # Use a quick min distance between segment and AABB:
    # If intersects -> negative (penetration)
    hit, t_entry = _segment_aabb_first_hit(x0, y0, x1, y1, (xmin, xmax, ymin, ymax))
    if hit:
        return -1e-6  # treat as penetrating/touching

    # Otherwise compute min distance from segment to box
    # Sample endpoints and closest points to box
    def point_aabb_dist(px, py):
        cx = min(max(px, xmin), xmax)
        cy = min(max(py, ymin), ymax)
        dx, dy = px - cx, py - cy
        return math.hypot(dx, dy)

    # also check the closest point on segment to the box corners
    d = min(point_aabb_dist(x0, y0), point_aabb_dist(x1, y1))
    return d
