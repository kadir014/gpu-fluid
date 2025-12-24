from array import array
from struct import unpack
from time import perf_counter
from math import ceil
from random import randint

import pygame
import moderngl


WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
MAX_FPS = 60
ITERATIONS = 10

# Particle configuration
N = 1_000_000
SIZE = 2.0 # pixel
ALPHA = 0.25 # [0, 1]


pygame.init()
pygame.display.set_mode(WINDOW_SIZE, flags=pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.Clock()
is_running = True


context = moderngl.create_context(require=460)
context.enable(moderngl.BLEND)

print()
print("GPU:            ", context.info["GL_RENDERER"])
print("GL:             ", context.info["GL_VERSION"])
print("Max work groups:", context.info["GL_MAX_COMPUTE_WORK_GROUP_COUNT"])
print("Thread axes:    ", context.info["GL_MAX_COMPUTE_WORK_GROUP_SIZE"])
print("Max threads:    ", context.info["GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS"])
print()
print("Max SSBO bindings:     ", context.info["GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS"])
print("Max uniform components:", context.info["GL_MAX_COMPUTE_UNIFORM_COMPONENTS"], "floats")
print("Max uniform bindings:  ", context.info["GL_MAX_COMPUTE_UNIFORM_BLOCKS"])
print("Max uniform budget:    ", context.info["GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS"], "floats")
print("Max atomic counters:   ", context.info["GL_MAX_COMPUTE_ATOMIC_COUNTERS"])
print("Max atomic buffers:    ", context.info["GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS"])
print("Max images:            ", context.info["GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS"])


class ScreenQuad:
    def __init__(self) -> None:
        base_vertex_shader = """
        #version 460

        in vec2 in_position;
        in vec2 in_uv;

        out vec2 v_uv;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);

            v_uv = in_uv;
        }
        """

        base_fragment_shader = """
        #version 460

        in vec2 v_uv;

        out vec4 f_color;

        uniform sampler2D s_texture;

        void main() {
            vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
            f_color = texture(s_texture, uv).bgra;
        }
        """

        self._vbo = self.create_buffer_object([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        self._uvbo = self.create_buffer_object([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self._ibo = self.create_buffer_object([0, 1, 2, 1, 2, 3])

        self._program = context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=base_fragment_shader
        )
        self._program["s_texture"] = 0

        self._vao = context.vertex_array(
            self._program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._texture = context.texture(WINDOW_SIZE, 4)
        self._texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def create_buffer_object(self, data: list) -> moderngl.Buffer:
        dtype = "f" if isinstance(data[0], float) else "I"
        return context.buffer(array(dtype, data))
    
    def render_surface(self, surface: pygame.Surface) -> None:
        self._texture.write(surface.get_view("1"))
        self._texture.use(0)
        self._vao.render()

    def render_texture(self, texture: moderngl.Texture) -> None:
        texture.use(0)
        self._vao.render()


screenquad = ScreenQuad()
surface = pygame.Surface(WINDOW_SIZE)


compute_velocity_src = open("compute_velocity.comp", "r", encoding="utf-8").read()
compute_collision_src = open("compute_collision.comp", "r", encoding="utf-8").read()
compute_position_src = open("compute_position.comp", "r", encoding="utf-8").read()

# TODO: shader patcher
compute_velocity_src = compute_velocity_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_collision_src = compute_collision_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_position_src = compute_position_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")

compute_velocity = context.compute_shader(compute_velocity_src)
compute_collision = context.compute_shader(compute_collision_src)
compute_position = context.compute_shader(compute_position_src)

compute_collision["u_resolution"] = WINDOW_SIZE

texture = context.texture(WINDOW_SIZE, 4)
texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
texture.bind_to_image(0)



positions_alt = context.buffer(reserve=N * 2 * 4)
positions_alt.bind_to_storage_buffer(0)

positions_main = context.buffer(reserve=N * 2 * 4)
positions_main.bind_to_storage_buffer(1)

pos = []
for p in range(N):
    pos.append(randint(5, 500))
    pos.append(randint(5, 500))

positions_alt.write(array("f", pos))
positions_main.write(array("f", pos))


velocity_alt = context.buffer(reserve=N * 2 * 4)
velocity_alt.bind_to_storage_buffer(2)

velocity_main = context.buffer(reserve=N * 2 * 4)
velocity_main.bind_to_storage_buffer(3)

pos = []
for p in range(N):
    vel = pygame.Vector2(1, 0).rotate(randint(0, 360)) * float(randint(5, 15))
    pos.append(vel.x)
    pos.append(vel.y)

velocity_alt.write(array("f", pos))
velocity_main.write(array("f", pos))


base_vertex_shader = f"""
#version 460

in vec2 in_position;
in vec2 in_velocity;

out vec2 v_velocity;

void main() {{
    vec2 pos = vec2(in_position.x, {WINDOW_HEIGHT}.0 - in_position.y);
    vec2 ndc = (pos / vec2({WINDOW_WIDTH}, {WINDOW_HEIGHT}) - vec2(0.5)) * 2.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_velocity = in_velocity;
}}
"""

base_fragment_shader = f"""
#version 460

in vec2 v_velocity;

out vec4 f_color;

vec3 hsv2rgb(vec3 c) {{
    const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}}

void main() {{
    float speed = length(v_velocity) * 0.017;

    vec3 color = hsv2rgb(vec3(speed, 1.0, 1.0));

    f_color = vec4(color, {ALPHA});
}}
"""

_program = context.program(
    vertex_shader=base_vertex_shader,
    fragment_shader=base_fragment_shader
)

_vao = context.vertex_array(
    _program,
    (
        (positions_alt, "2f", "in_position"),
        (velocity_alt, "2f", "in_velocity"),
    ),
)


context.point_size = SIZE


while is_running:
    dt = clock.tick(MAX_FPS) * 0.001

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                is_running = False

    mouse = pygame.Vector2(*pygame.mouse.get_pos())

    start = perf_counter()

    compute_velocity["u_mouse"] = (mouse.x, mouse.y)
    for step in range(ITERATIONS):
        compute_velocity.run(ceil(N / 32.0), 1, 1)

        velocity_alt, velocity_main = velocity_main, velocity_alt
        velocity_alt.bind_to_storage_buffer(2)
        velocity_main.bind_to_storage_buffer(3)

        compute_collision.run(ceil(N / 32.0), 1, 1)

        velocity_alt, velocity_main = velocity_main, velocity_alt
        velocity_alt.bind_to_storage_buffer(2)
        velocity_main.bind_to_storage_buffer(3)

        compute_position.run(ceil(N / 32.0), 1, 1)

        positions_alt, positions_main = positions_main, positions_alt
        positions_alt.bind_to_storage_buffer(0)
        positions_main.bind_to_storage_buffer(1)

        velocity_alt, velocity_main = velocity_main, velocity_alt
        velocity_alt.bind_to_storage_buffer(2)
        velocity_main.bind_to_storage_buffer(3)

    elapsed0 = perf_counter() - start

    start = perf_counter()
    
    context.clear(0.0, 0.0, 0.0)
    _vao.render(moderngl.POINTS)
    pygame.display.flip()

    elapsed1 = perf_counter() - start

    pygame.display.set_caption(f"FPS: {round(clock.get_fps())}   Compute: {round(elapsed0*1000.0, 3)}ms   Render: {round(elapsed1*1000.0, 3)}ms")

context.release()
pygame.quit()