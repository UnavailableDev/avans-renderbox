# Particle simulation with OpenCL

## Libraries:
- pyOpenCL
- numpy
- matplotlib

Using venv:
```sh
python -m venv ./
```
```sh
source bin/activate
```

## THE PLAN

- [x] Particle comprised of position and velocity vectors
- [x] Particle list
- [x] Spawning particles with random position/velocity
- [x] Inter-particle collision
    - [x] simple elastic collision
    - [ ] research more collision solvers
- [x] simulation bounding box
- [ ] simulation window size
- [?] 3d simulation?
- [ ] object files
    - [ ] dynamically loadable
- [ ] find a OpenCL language server

