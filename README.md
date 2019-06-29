[![preview img with link to demo](https://i.imgur.com/8j7793v.png)](https://youtu.be/eMR_KqUL-Ag)

v0.1.0 build: [releases/tag/v0.1.0](https://github.com/gluyas/lemon/releases/tag/v0.1.0)  
Demo video: https://youtu.be/eMR_KqUL-Ag

# Lemon

In-development interactive toy, written in Rust. Exists for me to practice, develop, and showcase my game programming skills. Written as much from scratch as I feel feasible. Current external dependencies limited to:
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
- `-n` `maximum number of lemons` (default: 256)
- `-w` `window width` `-h` `window height` (default: 1280x720)
- `-v` [`true`|`false`] (enable vsync - default: false)

## Current Features
- Simple rigidbody dynamics simulation
  - Ability to step forward and back through physics frames for debugging
  - Friction and drag models
- Analytically defined lemon model, including
  - Surface mesh
  - Physical properties
    - Mass; Moment of inertia tensor
  - Collision detection algorithm
    - Currently only supports lemon xy-plane testing
  - Graph visualisation at [desmos.com](https://www.desmos.com/calculator/z7ijifw8pc)
    - In 3D, the lemon is the self-intersection of a spindle torus. Collision detection tests are defined by an arc of the minor radius of the torus, offset from the centre of the lemon by the major radius of the torus.
- Graphics pipeline
  - Phong shading
  - Normal mapping (currently unused)

## Planned Features
- Multiple dynamic bodies
  - Requires lemon lemon collision test
  - More stable collision response for multiple collisions
- Fruit bowl object
  - Requires lemon hollow-sphere collision test
    - Non-convex, but should be feasible with knowledge of the lemon's shape
- Lemon growing on a branch animation
- Lemon variation
  - Size, shape, colour
  - Different citrus fruits
  - Already possible; particular values currently hardcoded
- Slicing lemons into independent bodies
  - Render lemon core
  - Determine physical properties of pieces
  - Collision detection
- Working UI loop / UX design
  - Gesture driven input
- Stylish graphics

## Sources Used
- http://chrishecker.com/Rigid_Body_Dynamics
- https://www.researchgate.net/publication/2173504
- http://realtimecollisiondetection.net
- https://www.gameenginebook.com
- https://gafferongames.com/post/collision_response_and_coulomb_friction
- https://www.sciencedirect.com/science/article/pii/S0307904X0200080X
- https://medium.com/@tglaiel/how-to-make-your-game-run-at-60fps-24c61210fe75
