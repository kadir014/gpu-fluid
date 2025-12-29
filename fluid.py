from array import array
from struct import unpack
from time import perf_counter
from math import ceil
from random import randint, uniform

import pygame
import moderngl


WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
MAX_FPS = 60
ITERATIONS = 10

N = 5
DOMAIN = pygame.FRect(0.0, 0.0, 10.0, 10.0)
DOMAIN_TO_SCREEN = 60.0
PARTICLE_RADIUS = 0.05
CELL_SIZE = PARTICLE_RADIUS
GRID_WIDTH = ceil(DOMAIN.width / CELL_SIZE)
GRID_HEIGHT = ceil(DOMAIN.height / CELL_SIZE)
GRID_CELL_COUNT = GRID_WIDTH * GRID_HEIGHT

print("Grid cells:", GRID_CELL_COUNT)

PARTICLE_ALPHA = 1.0
PARTICLE_SIZE = PARTICLE_RADIUS * DOMAIN_TO_SCREEN


pygame.init()
pygame.display.set_mode(WINDOW_SIZE, flags=pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.Clock()
is_running = True


context = moderngl.create_context(require=460)
context.enable(moderngl.BLEND)


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


build_grid_src = open("build_grid.comp", "r", encoding="utf-8").read()
sort0_src = open("sort_pass0.comp", "r", encoding="utf-8").read()
sort1_src = open("sort_pass1.comp", "r", encoding="utf-8").read()
sort2_src = open("sort_pass2.comp", "r", encoding="utf-8").read()
build_lut_src = open("build_lut.comp", "r", encoding="utf-8").read()

# TODO: shader patcher
build_grid_src = build_grid_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
build_grid_src = build_grid_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
build_grid_src = build_grid_src.replace("#define CELL_SIZE 1.0", f"#define CELL_SIZE {CELL_SIZE}")
build_grid_src = build_grid_src.replace("#define GRID_WIDTH 1", f"#define GRID_WIDTH {GRID_WIDTH}")
build_grid_src = build_grid_src.replace("#define GRID_HEIGHT 1", f"#define GRID_HEIGHT {GRID_HEIGHT}")
print(build_grid_src)
sort0_src = sort0_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort0_src = sort0_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
sort1_src = sort1_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort1_src = sort1_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
sort2_src = sort2_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort2_src = sort2_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
build_lut_src = build_lut_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
build_lut_src = build_lut_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")

build_grid_shader = context.compute_shader(build_grid_src)
sort0_shader = context.compute_shader(sort0_src)
sort1_shader = context.compute_shader(sort1_src)
sort2_shader = context.compute_shader(sort2_src)
build_lut_shader = context.compute_shader(build_lut_src)


entries = context.buffer(reserve=N * 4 * 2)
entries.bind_to_storage_buffer(0)

sorted_entries = context.buffer(reserve=N * 4 * 2)
sorted_entries.bind_to_storage_buffer(4)

cell_counts = context.buffer(reserve=GRID_CELL_COUNT * 4)
cell_counts.bind_to_storage_buffer(1)

cell_offsets = context.buffer(reserve=GRID_CELL_COUNT * 4)
cell_offsets.bind_to_storage_buffer(2)

cell_ranges = context.buffer(reserve=GRID_CELL_COUNT * 4 * 2)
cell_ranges.bind_to_storage_buffer(5)


# before sorting
# [0, 3, 0, 2, 2]

# cell counts
# [2, 0, 2, 1]

# cell_offsets
# [0, 2, 2, 4]

# after sorting
# [0, 0, 2, 2, 3]


states = []
# NOT INTERLEAVED, PACKED
for i in range(N):
    states.append(randint(0, GRID_CELL_COUNT-1))
    #states.append([0, 3, 0, 2, 2][i])
for i in range(N):
    states.append(0)
entries.write(array("L", states))
sorted_entries.clear()


cell_counts.clear()
cell_offsets.clear()




compute_velocity_src = open("compute_velocity.comp", "r", encoding="utf-8").read()
compute_neighbor_src = open("compute_neighbor.comp", "r", encoding="utf-8").read()
compute_collision_src = open("compute_collision.comp", "r", encoding="utf-8").read()
compute_position_src = open("compute_position.comp", "r", encoding="utf-8").read()

# TODO: shader patcher
compute_velocity_src = compute_velocity_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_neighbor_src = compute_neighbor_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_neighbor_src = compute_neighbor_src.replace("#define CELL_SIZE 1.0", f"#define CELL_SIZE {CELL_SIZE}")
compute_neighbor_src = compute_neighbor_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
compute_neighbor_src = compute_neighbor_src.replace("#define GRID_WIDTH 1", f"#define GRID_WIDTH {GRID_WIDTH}")
compute_neighbor_src = compute_neighbor_src.replace("#define GRID_HEIGHT 1", f"#define GRID_HEIGHT {GRID_HEIGHT}")
compute_collision_src = compute_collision_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_position_src = compute_position_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")

compute_velocity = context.compute_shader(compute_velocity_src)
compute_neighbor = context.compute_shader(compute_neighbor_src)
compute_collision = context.compute_shader(compute_collision_src)
compute_position = context.compute_shader(compute_position_src)

compute_collision["u_domain_min"] = DOMAIN.topleft
compute_collision["u_domain_max"] = DOMAIN.bottomright


particles_alt = context.buffer(reserve=N * 2 * 4 * 2)
particles_alt.bind_to_storage_buffer(6)

particles_main = context.buffer(reserve=N * 2 * 4 * 2)
particles_main.bind_to_storage_buffer(7)

states = []
for p in range(N):
    states.append(uniform(DOMAIN.left+0.1, DOMAIN.right-0.1))
    states.append(uniform(DOMAIN.top+0.1, DOMAIN.bottom-0.1))

for p in range(N):
    vel = pygame.Vector2(1, 0).rotate(randint(0, 360)) * float(randint(5, 15)) * 0
    states.append(vel.x)
    states.append(vel.y)

particles_alt.write(array("f", states))
particles_main.write(array("f", states))


base_vertex_shader = f"""
#version 460

in vec2 in_position;
in vec2 in_velocity;

out vec2 v_velocity;

void main() {{
    vec2 dpos = in_position * {DOMAIN_TO_SCREEN};
    vec2 pos = vec2(dpos.x, {WINDOW_HEIGHT}.0 - dpos.y);
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

    f_color = vec4(color, {PARTICLE_ALPHA});
}}
"""

_program = context.program(
    vertex_shader=base_vertex_shader,
    fragment_shader=base_fragment_shader
)

_alt_vao = context.vertex_array(
    _program,
    (
        (particles_alt, "2f", "in_position"),
    ),
)
_alt_vao.bind(1, "f", particles_alt, "2f", N * 2 * 4)

_main_vao = context.vertex_array(
    _program,
    (
        (particles_main, "2f", "in_position"),
    ),
)
_main_vao.bind(1, "f", particles_main, "2f", N * 2 * 4)


context.point_size = PARTICLE_SIZE



frame = 0
while is_running:
    dt = clock.tick(MAX_FPS) * 0.001

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                is_running = False

    mouse = pygame.Vector2(*pygame.mouse.get_pos())
    pmouse = mouse / DOMAIN_TO_SCREEN


    # print()
    # print("before sorting:")
    # data = entries.read()
    # d = unpack(f"{N*2}I", data)
    # print(d[:N])

    start = perf_counter()

    cell_ranges.clear()
    build_grid_shader.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()

    # TODO: BUNLAR NIYE ÇOK ZAMAN YIYOR
    cell_counts.clear()
    cell_offsets.clear()

    # Histogram
    sort0_shader.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()

    # print("cell_counts:")
    # data = cell_counts.read()
    # d = unpack(f"{GRID_CELL_COUNT}I", data)
    # print(d)

    # Prefix Sum
    # EXACTLY ONE WORK GROUP!
    sort1_shader.run(1, 1, 1)
    context.memory_barrier()

    # print("cell_offsets:")
    # data = cell_offsets.read()
    # d = unpack(f"{GRID_CELL_COUNT}I", data)
    # print(d)

    # Scatter
    sort2_shader.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()


    build_lut_shader.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()

    current_is_main = False

    compute_velocity["u_mouse"] = (pmouse.x, pmouse.y)
    compute_velocity.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()
    particles_alt, particles_main = particles_main, particles_alt
    particles_alt.bind_to_storage_buffer(6)
    particles_main.bind_to_storage_buffer(7)
    current_is_main = not current_is_main

    # compute_neighbor.run(ceil(N / 32.0), 1, 1)
    # context.memory_barrier()
    # particles_alt, particles_main = particles_main, particles_alt
    # particles_alt.bind_to_storage_buffer(6)
    # particles_main.bind_to_storage_buffer(7)

    compute_collision.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()
    particles_alt, particles_main = particles_main, particles_alt
    particles_alt.bind_to_storage_buffer(6)
    particles_main.bind_to_storage_buffer(7)
    current_is_main = not current_is_main

    compute_position.run(ceil(N / 32.0), 1, 1)
    context.memory_barrier()
    particles_alt, particles_main = particles_main, particles_alt
    particles_alt.bind_to_storage_buffer(6)
    particles_main.bind_to_storage_buffer(7)
    current_is_main = not current_is_main

    #particles_alt, particles_main = particles_main, particles_alt
    #particles_alt.bind_to_storage_buffer(6)
    #particles_main.bind_to_storage_buffer(7)
    #particle_dispatches += 1


    elapsed0 = perf_counter() - start

    # print("after sorting:")
    # data = sorted_entries.read()
    # d = unpack(f"{N*2}I", data)
    # print(d[:N])


    #particles_alt, particles_main = particles_main, particles_alt
    #particles_alt.bind_to_storage_buffer(0)
    #particles_main.bind_to_storage_buffer(1)


    start = perf_counter()

    context.clear(0.0, 0.0, 0.0)

    if frame % 2 == 0 and current_is_main:
        _main_vao.render(moderngl.POINTS, vertices=N)
    else:
        _alt_vao.render(moderngl.POINTS, vertices=N)
    frame += 1

    pygame.display.flip()

    elapsed1 = perf_counter() - start

    pygame.display.set_caption(f"FPS: {round(clock.get_fps())}   Compute: {round(elapsed0*1000.0, 3)}ms   Render: {round(elapsed1*1000.0, 3)}ms")

context.release()
pygame.quit()