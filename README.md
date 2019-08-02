[![preview img with link to demo](https://i.imgur.com/4rFLvma.png)](https://gluyas.github.io/lemon/)

Play here: [gluyas.github.io/lemon](https://gluyas.github.io/lemon/)  
Main branch: [github.com/gluyas/lemon](https://github.com/gluyas/lemon)

# Lemon (Web Version)

This is a reduced-feature fork of the program that targets WebAssembly and WebGL via Rust's Emscripten backend.
Cut features include fast-forward/rewind, performance histogram, wireframes, and access to program settings.
Some controls are different.
There are also some slight reductions to the graphics to get around the limitations of OpenGL ES 3, which Emscripten provides an emulation layer for.

### Controls

| Input | Action |
| :---: | ------ |
| Mouse Drag | Orbit camera |
| `↑` `↓` | Zoom camera |
| Mouse Click | Set camera look/follow target |
| Mouse Click + Drag | Grab and move lemon |
| `Space` | Reset lemon/camera position |
| `Ctrl` + `Space` | Create new lemon |
| `Ctrl` + `↑` `↓` | Edit lemon scale |
| `Alt` + `↑` `↓` | Edit lemon sagitta (width) |
| `Esc` | Pause/Unpause simulation |
| `R` | Reset simulation |

| Toggle | Debug Visualisation |
| :----: | ------------------- |
| `A` | Model-space transform axes |
| `S` | Torus cross-section |
| `Z` | Lemon-Lemon collision detection constructions |
| `X` | Lemon-Floor collision detection constructions |
| `C` | Collision reaction and friction impulses |
| `V` | Linear and angular velocity vectors |
| `B` | Bounding volumes |
| `L` | Party mode |
