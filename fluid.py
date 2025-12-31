from array import array
from struct import unpack
from time import perf_counter, time
from math import ceil
from random import randint, uniform

import pygame
import moderngl

from miniprofiler import Profiler


WINDOW_SIZE = WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
MAX_FPS = 200
ITERATIONS = 10

ENABLE_VISCOSITY = False

N = 100_000
CURRENT_N = N // 2
DOMAIN = pygame.FRect(0.0, 0.0, 128.0, 72.0)
DOMAIN_TO_SCREEN = 10.0
PARTICLE_RADIUS = 0.7 * 1.0
CELL_SIZE = PARTICLE_RADIUS
GRID_WIDTH = ceil(DOMAIN.width / CELL_SIZE)
GRID_HEIGHT = ceil(DOMAIN.height / CELL_SIZE)
GRID_CELL_COUNT = GRID_WIDTH * GRID_HEIGHT

print("Grid cells:", GRID_CELL_COUNT)

PARTICLE_ALPHA = 1.0
PARTICLE_SIZE = PARTICLE_RADIUS * DOMAIN_TO_SCREEN * 0.5 * 2.0


profiler = Profiler()
profiler.register("frame")
profiler.register("compute")
profiler.register("render")

profiler.register("grid")
profiler.register("physics")

profiler.register("histogram")
profiler.register("prefixsum")
profiler.register("scatter")

profiler.register("forces")
profiler.register("predict")
profiler.register("ddr")
profiler.register("collisions")
profiler.register("position")


pygame.init()
pygame.display.set_mode(WINDOW_SIZE, flags=pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.Clock()
is_running = True


context = moderngl.create_context(require=460)
context.enable(moderngl.BLEND)
#context.enable(moderngl.PROGRAM_POINT_SIZE)
context.enable_direct(0x8861)


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
clear_src = open("clear.comp", "r", encoding="utf-8").read()

# TODO: shader patcher
build_grid_src = build_grid_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
build_grid_src = build_grid_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
build_grid_src = build_grid_src.replace("#define CELL_SIZE 1.0", f"#define CELL_SIZE {CELL_SIZE}")
build_grid_src = build_grid_src.replace("#define GRID_WIDTH 1", f"#define GRID_WIDTH {GRID_WIDTH}")
build_grid_src = build_grid_src.replace("#define GRID_HEIGHT 1", f"#define GRID_HEIGHT {GRID_HEIGHT}")
sort0_src = sort0_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort0_src = sort0_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
sort1_src = sort1_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort1_src = sort1_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
sort2_src = sort2_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
sort2_src = sort2_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
build_lut_src = build_lut_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
build_lut_src = build_lut_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
clear_src = clear_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
clear_src = clear_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")

build_grid_shader = context.compute_shader(build_grid_src)
sort0_shader = context.compute_shader(sort0_src)
sort1_shader = context.compute_shader(sort1_src)
sort2_shader = context.compute_shader(sort2_src)
build_lut_shader = context.compute_shader(build_lut_src)
clear_shader = context.compute_shader(clear_src)


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
entries.write(array("I", states))
sorted_entries.clear()


cell_counts.clear()
cell_offsets.clear()



spawn_src = open("spawn.comp", "r", encoding="utf-8").read()
compute_velocity_src = open("compute_velocity.comp", "r", encoding="utf-8").read()
compute_viscosity_src = open("compute_viscosity.comp", "r", encoding="utf-8").read()
compute_predict_src = open("compute_predict.comp", "r", encoding="utf-8").read()
compute_neighbor_src = open("compute_neighbor.comp", "r", encoding="utf-8").read()
compute_collision_src = open("compute_collision.comp", "r", encoding="utf-8").read()
compute_position_src = open("compute_position.comp", "r", encoding="utf-8").read()

# TODO: shader patcher
spawn_src = spawn_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_velocity_src = compute_velocity_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_viscosity_src = compute_viscosity_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_viscosity_src = compute_viscosity_src.replace("#define CELL_SIZE 1.0", f"#define CELL_SIZE {CELL_SIZE}")
compute_viscosity_src = compute_viscosity_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
compute_viscosity_src = compute_viscosity_src.replace("#define GRID_WIDTH 1", f"#define GRID_WIDTH {GRID_WIDTH}")
compute_viscosity_src = compute_viscosity_src.replace("#define GRID_HEIGHT 1", f"#define GRID_HEIGHT {GRID_HEIGHT}")
compute_predict_src = compute_predict_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_neighbor_src = compute_neighbor_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_neighbor_src = compute_neighbor_src.replace("#define CELL_SIZE 1.0", f"#define CELL_SIZE {CELL_SIZE}")
compute_neighbor_src = compute_neighbor_src.replace("#define CELL_N 1", f"#define CELL_N {GRID_CELL_COUNT}")
compute_neighbor_src = compute_neighbor_src.replace("#define GRID_WIDTH 1", f"#define GRID_WIDTH {GRID_WIDTH}")
compute_neighbor_src = compute_neighbor_src.replace("#define GRID_HEIGHT 1", f"#define GRID_HEIGHT {GRID_HEIGHT}")
compute_collision_src = compute_collision_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")
compute_position_src = compute_position_src.replace("#define PARTICLE_N 1", f"#define PARTICLE_N {N}")

spawn_shader = context.compute_shader(spawn_src)
compute_velocity = context.compute_shader(compute_velocity_src)
compute_viscosity = context.compute_shader(compute_viscosity_src)
compute_predict = context.compute_shader(compute_predict_src)
compute_neighbor = context.compute_shader(compute_neighbor_src)
compute_collision = context.compute_shader(compute_collision_src)
compute_position = context.compute_shader(compute_position_src)

compute_collision["u_domain_min"] = DOMAIN.topleft
compute_collision["u_domain_max"] = DOMAIN.bottomright


particles_alt = context.buffer(reserve=N * 2 * 4 * 3)
particles_alt.bind_to_storage_buffer(6)

particles_main = context.buffer(reserve=N * 2 * 4 * 3)
particles_main.bind_to_storage_buffer(7)

states = []
pos = []
for p in range(N):
    x = uniform(DOMAIN.left+0.1, DOMAIN.right * 0.6)
    y = uniform(DOMAIN.top+0.1, DOMAIN.bottom-0.1)
    states.append(x)
    states.append(y)
    pos.append((x, y))

for p in range(N):
    states.append(0.0)
    states.append(0.0)

for p in range(N):
    vel = pygame.Vector2(1, 0).rotate(randint(0, 360)) * float(randint(5, 15))
    states.append(vel.x)
    states.append(vel.y)

particles_alt.write(array("f", states))
particles_main.write(array("f", states))


fluid_alt = context.buffer(reserve=N * 1 * 4 * 11)
fluid_alt.bind_to_storage_buffer(8)

fluid_main = context.buffer(reserve=N * 1 * 4 * 11)
fluid_main.bind_to_storage_buffer(9)

density_rest = 4.0
K = 120.0 * 4
K_near = 200.0 * 4
sigma = 90.0 * 1.0
beta = 80.0 * 1.0
fluid_state = (
    0.0,
    0.0,
    density_rest,
    K,
    K_near,
    0.0,
    0.0,
    sigma,
    beta
)
states = []
for i in range(9):
    for p in range(N):
        states.append(fluid_state[i])

uv_scale = 0.1
uv_scale = 0.01
for p in range(N):
    states.append(pos[p][0] * uv_scale)
    states.append(pos[p][1] * uv_scale)

fluid_alt.write(array("f", states))
fluid_main.write(array("f", states))


base_vertex_shader = f"""
#version 460

in vec2 in_position;
in vec2 in_velocity;

out vec2 v_velocity;
out vec2 v_uv;
out flat uint v_id;

void main() {{
    vec2 dpos = in_position * {DOMAIN_TO_SCREEN};
    vec2 pos = vec2(dpos.x, {WINDOW_HEIGHT}.0 - dpos.y);
    vec2 ndc = (pos / vec2({WINDOW_WIDTH}, {WINDOW_HEIGHT}) - vec2(0.5)) * 2.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_velocity = in_velocity;
    v_uv = ndc;
    v_id = gl_VertexID;
}}
"""

base_fragment_shader = f"""
#version 460

in vec2 v_velocity;
in vec2 v_uv;
in flat uint v_id;

out vec4 f_color;

uniform sampler2D s_texture;

uniform float u_dt;
uniform float u_time;

layout(std430, binding = 8) readonly buffer fluid {{
    float density[{N}];
    float density_near[{N}];
    float density_rest[{N}];

    float stiffness[{N}];
    float stiffness_near[{N}];

    float pressure[{N}];
    float pressure_near[{N}];

    float viscosity_sigma[{N}];
    float viscosity_beta[{N}];

    vec2 uv_in[{N}];
}};

vec3 hsv2rgb(vec3 c) {{
    const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}}

void main() {{
    vec2 uv = gl_PointCoord;
    float delta = length(uv - 0.5) - 0.5;
    //float alpha = smoothstep(0.0, 1.0, delta);
    //if (alpha < 0.01) discard;
    if (delta > 0.0) discard;
    float alpha = abs(delta) * 2.0;


    vec2 disp = v_velocity * u_dt;

    float speed = length(v_velocity);

    //vec3 color = hsv2rgb(vec3(speed, 1.0, 1.0));

    //vec2 offset = disp * 2.0;
    
    //vec3 color = texture(s_texture, v_uv + offset).rgb;

    //color += speed * 6.0;

    uv = uv_in[v_id];
    vec3 textured = texture(s_texture, uv + u_time).rgb;
    
    vec3 flat_color = vec3(1.0, 0.0, 0.0);
    flat_color = vec3(0.105, 0.792, 1.0);
    flat_color = vec3(0.031372, 0.48235, 1.0);

    float m = clamp(speed * 0.09, 0.0, 1.0);
    vec3 final_texture = mix(flat_color, textured, m);
    //final_texture = textured;

    // Velocity hue tint towards gree
    float tint = speed * 0.01;
    final_texture += vec3(0.0, tint, 0.0);

    
    // Pressure hue tint towards purple
    float p = clamp((pressure[v_id]) * 0.00017, 0.0, 1.0);
    p = smoothstep(0.0, 1.0, p);
    


    final_texture = clamp(final_texture, 0.0, 1.0);


    //float d = clamp(pressure_near[v_id] * 0.0006, 0.0, 1.0);
    float d = clamp((pressure_near[v_id]) * 0.00045, 0.0, 1.0);
    d = d;
    d = clamp(d, 0.0, 1.0);
    vec3 color = mix(vec3(1.0), final_texture, d);


    //color = vec3(p, 0.0, 0.0);


    //color = vec3(gl_PointCoord, 1.0);


    f_color = vec4(color, alpha);
}}
"""

_program = context.program(
    vertex_shader=base_vertex_shader,
    fragment_shader=base_fragment_shader
)

debug_surf = pygame.image.load("water_blue.png")
debug_tex = context.texture(debug_surf.size, 3, pygame.image.tobytes(debug_surf, "RGB"))

_alt_vao = context.vertex_array(
    _program,
    (
        (particles_alt, "2f", "in_position"),
    ),
)
_alt_vao.bind(1, "f", particles_alt, "2f", N * 2 * 4 * 2)

_main_vao = context.vertex_array(
    _program,
    (
        (particles_main, "2f", "in_position"),
    ),
)
_main_vao.bind(1, "f", particles_main, "2f", N * 2 * 4 * 2)


context.point_size = PARTICLE_SIZE

compute_velocity["u_gravity"] = (0.0, 10.0 * 2.5)


last_log = time()

frame = 0
while is_running:
    with profiler.profile("frame"):
        dt = clock.tick(MAX_FPS) * 0.001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    is_running = False

        mouse = pygame.Vector2(*pygame.mouse.get_pos())
        pmouse = mouse / DOMAIN_TO_SCREEN

        with profiler.profile("compute"):
            if pygame.mouse.get_pressed()[2] and CURRENT_N < N:
                AMOUNT = 100
                CURRENT_N += AMOUNT
                if CURRENT_N > N: CURRENT_N = N

                spawn_shader["u_mouse"] = pmouse
                spawn_shader["u_amount"] = AMOUNT
                spawn_shader["u_current_n"] = CURRENT_N

                spawn_shader.run(ceil(N / 32.0), 1, 1)
                context.memory_barrier()
                particles_alt, particles_main = particles_main, particles_alt
                particles_alt.bind_to_storage_buffer(6)
                particles_main.bind_to_storage_buffer(7)

            substeps = 3
            hertz = 120

            sim_dt = 1.0 / hertz / float(substeps)
            sim_inv_dt = 1.0 / sim_dt
            compute_velocity["u_dt"] = sim_dt
            compute_viscosity["u_dt"] = sim_dt
            compute_predict["u_dt"] = sim_dt
            compute_position["u_inv_dt"] = sim_inv_dt
            compute_neighbor["u_dt"] = sim_dt
            #_program["u_dt"] = sim_dt
            #_program["u_time"] = float(pygame.time.get_ticks()) * 0.000

            compute_velocity["u_current_n"] = CURRENT_N
            compute_viscosity["u_current_n"] = CURRENT_N
            compute_predict["u_current_n"] = CURRENT_N
            compute_neighbor["u_current_n"] = CURRENT_N
            compute_position["u_current_n"] = CURRENT_N
            compute_collision["u_current_n"] = CURRENT_N

            for i in range(substeps):
                with profiler.profile("grid"):
                    with profiler.profile("histogram"):
                        # TODO: BUNLAR NIYE ÇOK ZAMAN YIYOR
                        #cell_ranges.clear()
                        #cell_counts.clear()
                        #cell_offsets.clear()

                        clear_shader.run(ceil(GRID_CELL_COUNT / 32.0), 1, 1)
                        context.memory_barrier()

                        build_grid_shader.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()

                        # Histogram
                        sort0_shader.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()

                        # print("cell_counts:")
                        # data = cell_counts.read()
                        # d = unpack(f"{GRID_CELL_COUNT}I", data)
                        # print(d)

                    with profiler.profile("prefixsum"):
                        # Prefix Sum
                        # EXACTLY ONE WORK GROUP!
                        sort1_shader.run(1, 1, 1)
                        context.memory_barrier()

                        # print("cell_offsets:")
                        # data = cell_offsets.read()
                        # d = unpack(f"{GRID_CELL_COUNT}I", data)
                        # print(d)

                    with profiler.profile("scatter"):
                        # Scatter
                        sort2_shader.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()


                        build_lut_shader.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()


                with profiler.profile("physics"):
                    current_is_main = False
                    with profiler.profile("forces"):
                        if (pygame.mouse.get_pressed()[0]):
                            compute_velocity["u_mouse"] = (pmouse.x, pmouse.y)
                        else:
                            compute_velocity["u_mouse"] = (5000, 5000)
                        compute_velocity.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()
                        particles_alt, particles_main = particles_main, particles_alt
                        particles_alt.bind_to_storage_buffer(6)
                        particles_main.bind_to_storage_buffer(7)
                        current_is_main = not current_is_main

                    with profiler.profile("predict"):
                        if ENABLE_VISCOSITY:
                            compute_viscosity.run(ceil(N / 32.0), 1, 1)
                            context.memory_barrier()
                            particles_alt, particles_main = particles_main, particles_alt
                            particles_alt.bind_to_storage_buffer(6)
                            particles_main.bind_to_storage_buffer(7)
                            current_is_main = not current_is_main

                            # fluid_alt, fluid_main = fluid_main, fluid_alt
                            # fluid_alt.bind_to_storage_buffer(8)
                            # fluid_main.bind_to_storage_buffer(9)

                        compute_predict.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()
                        particles_alt, particles_main = particles_main, particles_alt
                        particles_alt.bind_to_storage_buffer(6)
                        particles_main.bind_to_storage_buffer(7)
                        current_is_main = not current_is_main

                    with profiler.profile("ddr"):
                        compute_neighbor.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()
                        particles_alt, particles_main = particles_main, particles_alt
                        particles_alt.bind_to_storage_buffer(6)
                        particles_main.bind_to_storage_buffer(7)
                        current_is_main = not current_is_main

                        fluid_alt, fluid_main = fluid_main, fluid_alt
                        fluid_alt.bind_to_storage_buffer(8)
                        fluid_main.bind_to_storage_buffer(9)

                    with profiler.profile("collisions"):
                        compute_collision.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()
                        particles_alt, particles_main = particles_main, particles_alt
                        particles_alt.bind_to_storage_buffer(6)
                        particles_main.bind_to_storage_buffer(7)
                        current_is_main = not current_is_main

                    with profiler.profile("position"):
                        compute_position.run(ceil(N / 32.0), 1, 1)
                        context.memory_barrier()
                        particles_alt, particles_main = particles_main, particles_alt
                        particles_alt.bind_to_storage_buffer(6)
                        particles_main.bind_to_storage_buffer(7)
                        current_is_main = not current_is_main

        with profiler.profile("render"):
            context.clear(0.0, 0.0, 0.0)

            debug_tex.use(0)

            if frame % 2 == 0 and current_is_main:
                _main_vao.render(moderngl.POINTS, vertices=CURRENT_N)
            else:
                _alt_vao.render(moderngl.POINTS, vertices=CURRENT_N)
            frame += 1

            pygame.display.flip()

    pygame.display.set_caption(f"GPU Fluid  -  FPS: {round(clock.get_fps())}  Particles: {CURRENT_N:,}/{N:,}  Cells: {GRID_CELL_COUNT:,}  Compute: {round(profiler['compute'].avg * 1000, 3)}ms   Render: {round(profiler['render'].avg * 1000, 3)}ms")

    if time() - last_log > 1.0:
        last_log = time()

        print("\033c", end="")
        print("Average profiling of last second in milliseconds")
        print("================================================")
        print(f"Frame:   {round(profiler['frame'].avg * 1000.0, 3)}ms")
        print(f"Compute: {round(profiler['compute'].avg * 1000.0, 3)}ms")
        print(f"Render:  {round(profiler['render'].avg * 1000.0, 3)}ms")
        print()
        print(f"Grid: {round(profiler['grid'].avg * 1000.0, 3)}ms")
        print(f"-  Histogram: {round(profiler['histogram'].avg * 1000.0, 3)}ms")
        print(f"-  Prefixsum: {round(profiler['prefixsum'].avg * 1000.0, 3)}ms")
        print(f"-  Scatter:   {round(profiler['scatter'].avg * 1000.0, 3)}ms")
        print()
        print(f"Physics: {round(profiler['physics'].avg * 1000.0, 3)}ms")
        print(f"-  Forces:    {round(profiler['forces'].avg * 1000.0, 3)}ms")
        print(f"-  Predict:   {round(profiler['predict'].avg * 1000.0, 3)}ms")
        print(f"-  DDR:       {round(profiler['ddr'].avg * 1000.0, 3)}ms")
        print(f"-  Collision: {round(profiler['collisions'].avg * 1000.0, 3)}ms")
        print(f"-  Position:  {round(profiler['position'].avg * 1000.0, 3)}ms")

context.release()
pygame.quit()