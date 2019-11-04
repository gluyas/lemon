[![preview img with link to demo](https://i.imgur.com/4rFLvma.png)](https://youtu.be/zo6fOU6ERJ0)

v0.2.1 build: [releases/tag/v0.2.1](https://github.com/gluyas/lemon/releases/tag/v0.2.1)  
Demo video: https://youtu.be/zo6fOU6ERJ0

# Lemon

Work-in-progress interactive toy. Written (mostly) from scratch, in Rust. Current top-level dependencies limited to:
- OpenGL bindings ([gl_generator](https://crates.io/crates/gl_generator))
- Windowing, user input handling ([glutin](https://crates.io/crates/glutin))
- 3D math types ([cgmath](https://crates.io/crates/cgmath))
- Rust standard library; other lightweight, de-facto-standard crates

### Controls

| Input | Action |
| :---: | ------ |
| Mouse Drag | Orbit camera |
| Mouse Scroll | Zoom camera |
| Mouse Click | Set camera look/follow target |
| Mouse Click + Drag | Grab and move lemon |
| `Space` | Reset lemon/camera position |
| `Ctrl` + `Space` | Create new lemon |
| `Ctrl` + Scroll | Edit lemon scale |
| `Alt` + Scroll | Edit lemon sagitta (width) |
| `Esc` | Pause/Unpause simulation |
| (`Ctrl` +) `←` `→` | Step through physics frames while paused |
| `Ctrl` + `Shift` + `R` | Reset simulation |

| Toggle | Debug Visualisation |
| :----: | ------------------- |
| `Q` | Work-in-progress SDF renderer |
| `A` | Model-space transform axes |
| `S` | Torus cross-section |
| `H` | Performance histogram |
| `Z` | Lemon-Lemon collision detection constructions |
| `X` | Lemon-Floor collision detection constructions |
| `C` | Collision reaction and friction impulses |
| `V` | Linear and angular velocity vectors |
| `B` | Bounding volumes |
| `W` | Wireframes |
| `L` | Party mode |

| Key | Default | Modifiers | Program Setting |
| :-: | :-----: | :-------: | --------------- |
| `J` | Enable | `Ctrl`: Disable | Histogram data logging |
| `T` | Sleep-then-spin | `Ctrl`: Spin only | Frame timing method |

### Command line options
These are all set to run on a very low-end machine; crank them up!
- `-n` `maximum number of lemons` (default: 256)
- `-r` [`360`|`480`|`720`|`1080`|`1440`|`2160`] (common resolutions - default: 720)
- `-w` `custom window width` `-h` `custom window height` (default: 1280x720)
- `-v` [`true`|`false`] (enable vsync - default: false)
- `-m` `multisampling level` (default: 2)

## Current Features
- Impulse-based rigidbody physics simulation
  - Multiple bodies
  - Friction and drag models
- Analytically defined lemon model
  - 3D mesh, normals
    - Dynamic surface parametisation
  - Physical properties
    - Mass; Moment of inertia tensor
  - Collision detection procedures (with some help from numeric methods)
    - Lemon-lemon, lemon-plane collisions
    - Capsule based bounding-volumes
    - Raycasting
- Mouse-driven input
- Debugging utilities
  - Pause, step, rewind, and fast-forward functionality
  - Simple profiling histogram and API
  - Simple 3D line and pixel drawing APIs
    - Supports visualisations of complex procedures in the program
- Graphics pipeline
  - Phong shading
  - Normal mapping (deprecated)

## Todo / Wishlist
- Actually nice graphics
  - Some cool art style that is tractable for me to implement
  - Procedural surface detail
    - Difficult due to UV map pinching at poles
    - Screenspace-based alternatives?
  - Other classes of citrus fruits: colour, shape, size, texture, etc.
  - Scene composition
    - Citrus tree which fruit grow and fall from
    - Fruit bowl, plates, picnic blanket, etc.
- Slicing lemons
  - Render lemon core along plane of slicing
  - Dynamic slice models
    - Mesh creation
    - Collision algorithm and physical properties
- Better physics
  - Constraint solver: currently only resolves collision pairs in a fixed order
  - Numerical integrator, especially for angular effects
    - Additionally: continuous collision detection with respect to rotation; The current implementation really struggles with narrow bodies


## Sources Used
- http://chrishecker.com/Rigid_Body_Dynamics
- https://www.researchgate.net/publication/2173504
- http://realtimecollisiondetection.net
- https://www.gameenginebook.com
- https://gafferongames.com/post/collision_response_and_coulomb_friction
- https://www.sciencedirect.com/science/article/pii/S0307904X0200080X
