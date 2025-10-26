import os
import sys
import math
import numpy as np

# --- macOS compatibility fix for pyglet ---
if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"

# --- Import pyglet safely ---
try:
    import pyglet
except ImportError:
    raise ImportError("Pyglet is required for rendering. Please install it via `pip install pyglet`.")

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        "Error importing OpenGL from pyglet. "
        "This likely occurred because pyglet installed the wrong architecture binaries."
    )

RAD2DEG = 57.29577951308232


# ============================================================================ #
#                                Utility Functions                             #
# ============================================================================ #

def get_display(spec):
    """
    Converts a display specification (e.g., ':0') into a pyglet Display object.
    Pyglet only supports multiple displays on Linux.

    Args:
        spec (str or None): Display specification string.

    Returns:
        pyglet.canvas.Display or None
    """
    if spec is None:
        return None
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise ValueError(
            f"Invalid display specification: {spec}. Must be a string like ':0' or None."
        )


# ============================================================================ #
#                                   Viewer                                     #
# ============================================================================ #

class Viewer:
    def __init__(self, width, height, display=None, visible=True):
        display = get_display(display)
        self.width = width
        self.height = height

        self.window = pyglet.window.Window(
            width=width, height=height, display=display, vsync=False, resizable=True, visible=visible
        )
        self.window.on_close = self.window_closed_by_user
        self.window.push_handlers(self)

        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        """Safely close the rendering window."""
        if self.window:
            self.window.close()
            self.window = None

    def window_closed_by_user(self):
        """Ensure proper cleanup when the user closes the window."""
        self.close()

    def on_resize(self, width, height):
        """Handle window resizing."""
        self.width = width
        self.height = height

    def set_bounds(self, left, right, bottom, top):
        """Set the visible bounds of the window."""
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley)
        )

    def add_geom(self, geom):
        """Add a persistent geometry to be rendered each frame."""
        self.geoms.append(geom)

    def add_onetime(self, geom):
        """Add a geometry to be rendered once."""
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        """
        Render the current scene.
        Returns the RGB image array if `return_rgb_array=True`.
        """

        # --- Safety Check 1: Handle rendering after window close ---
        if self.window is None or self.window.has_exit:
            if return_rgb_array:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return None

        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()  # Process on_close event if triggered

        # --- Safety Check 2: Recheck window status after event dispatch ---
        if self.window is None or self.window.has_exit:
            if return_rgb_array:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return None

        glViewport(0, 0, self.width, self.height)
        self.transform.enable()

        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()

        self.transform.disable()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr

    # --- Drawing Utilities ---

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_mouse_coords(self):
        return (self.mouse_x, self.mouse_y)


# ============================================================================ #
#                            Geometry and Attributes                           #
# ============================================================================ #

class Geom:
    """Base geometry class."""
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)


class Attr:
    """Base attribute class."""
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    """Represents translation, rotation, and scaling transformations."""
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1.0, 1.0)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation.x, self.translation.y, 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale.x, self.scale.y, 1.0)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = Vec2(newx, newy)

    def set_rotation(self, newa):
        self.rotation = newa

    def set_scale(self, newx, newy):
        self.scale = Vec2(newx, newy)


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


# ============================================================================ #
#                                 Primitives                                   #
# ============================================================================ #

class Point(Geom):
    def render1(self):
        glBegin(GL_POINTS)
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
        glEnd()


class PolyLine(Geom):
    def __init__(self, v, close):
        super().__init__()
        self.v = v
        self.close = close

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
        glEnd()


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        super().__init__()
        self.start = start
        self.end = end

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        super().__init__()
        self.set_color(1.0, 1.0, 1.0)
        self.img = pyglet.image.load(fname)
        self.width = width
        self.height = height

    def render1(self):
        self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)


# ============================================================================ #
#                           Helper Functions                                   #
# ============================================================================ #

def make_circle(radius=10, res=30, filled=True):
    points = [(math.cos(2 * math.pi * i / res) * radius,
               math.sin(2 * math.pi * i / res) * radius) for i in range(res)]
    return FilledPolygon(points) if filled else PolyLine(points, True)


def make_polygon(v, filled=True):
    return FilledPolygon(v) if filled else PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.add_attr(LineWidth(attrs["linewidth"]))


# ============================================================================ #
#                             Simple Image Viewer                              #
# ============================================================================ #

class SimpleImageViewer:
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = display
        self.maxwidth = maxwidth

    def imshow(self, arr):
        if self.window is None:
            height, width, _ = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display, vsync=False, resizable=True
            )
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert arr.shape[0] == self.height and arr.shape[1] == self.width, \
            f"Image shape {arr.shape} does not match window size {(self.height, self.width)}."

        image = pyglet.image.ImageData(
            self.width, self.height, "RGB", arr.tobytes(), pitch=self.width * -3
        )
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


# ============================================================================ #
#                                   Vec2                                       #
# ============================================================================ #

class Vec2:
    """Simple 2D vector class."""
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vec2({self.x:.2f}, {self.y:.2f})"
