[![preview img with link to demo](https://i.imgur.com/8j7793v.png)](https://youtu.be/eMR_KqUL-Ag)

v0.1.0 build: [releases/tag/v0.1.0](https://github.com/gluyas/lemon/releases/tag/v0.1.0)  
Demo video: https://youtu.be/eMR_KqUL-Ag

# Lemon

In-development interactive toy, written in Rust. Exists for me to practice, develop, and showcase my game programming skills. Written as much from scratch as I feel feasible. Current external dependencies limited to:
- OpenGL bindings ([gl_generator](https://crates.io/crates/gl_generator))
- Windowing, user input handling ([glutin](https://crates.io/crates/glutin))
- 3D math types ([cgmath](https://crates.io/crates/cgmath))
- Image format processing ([png](https://crates.io/crates/png), [image](https://crates.io/crates/image))
- Rust standard library; other lightweight, de-facto-standard crates

### Controls
Drag and scroll to manipulate the camera

| Input | Action |
| :---: | ----- |
| `Space` | Reset current lemon |
| `Ctrl` + `Space` | Create new lemon |
| `Esc` | Pause/Unpause simulation |
| `←` `→` | Step through current lemon frames while paused |

| Toggle | Debug Visualisation |
| :--------: | ----------------------- |
| `X` | Collision detection constructions |
| `C` | Collision reaction and friction impulses |
| `V` | Linear and angular velocity vectors  |

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
