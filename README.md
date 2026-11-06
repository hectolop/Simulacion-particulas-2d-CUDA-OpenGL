# Simulación de Partículas — CUDA + OpenGL

Simulación en tiempo real de **80.000 partículas** con física de colisiones, implementada en CUDA y visualizada con OpenGL. El proyecto incluye una versión CPU multihilo para comparativa de rendimiento.

![Demo](demo.gif)

---

## Descripción

El programa simula partículas con gravedad, colisiones entre partículas y rebote con los bordes de la pantalla. Todo el cómputo ocurre en GPU mediante 4 kernels CUDA que se ejecutan cada frame:

| Kernel | Función |
|--------|---------|
| `updateParticles` | Física de gravedad, integración de Euler y rebote con bordes |
| `buildSpatialGrid` | Construcción de rejilla espacial para detección eficiente de colisiones |
| `resolveCollisions` | Detección y resolución de colisiones entre partículas |
| `renderParticles` | Escritura de píxeles en buffer de imagen en GPU |

La imagen resultante se transfiere a CPU vía `cudaMemcpy` y OpenGL la muestra en pantalla con `glDrawPixels`.

---

## Arquitectura

```
CPU HOST                          GPU DEVICE
────────────────                  ──────────────────────────────
Bucle principal (GLUT)      →     K1: updateParticles
                                  K2: buildSpatialGrid
                                  K3: resolveCollisions
                                  K4: renderParticles
                                        ↓ d_img
OpenGL (glDrawPixels) ←── cudaMemcpy D→H (~1.9 MB/frame)
```

Las partículas residen permanentemente en GPU desde el inicio hasta el fin del programa. La única transferencia CPU↔GPU en el bucle principal es la imagen final renderizada.

---

## Optimizaciones

- **Rejilla espacial 8×8px** — evita O(N²) comparaciones reduciendo colisiones a O(N×densidad local)
- **`atomicExch`** — inserción en rejilla sin condiciones de carrera entre hilos
- **`d_snapshot`** — copia DeviceToDevice antes de K3 para garantizar consistencia entre hilos
- **Partículas permanentes en GPU** — se suben una sola vez al inicio, nunca vuelven a CPU
- **Un hilo por partícula** en todos los kernels — máximo paralelismo

---

## Requisitos

- NVIDIA GPU con soporte CUDA
- CUDA Toolkit
- OpenGL + GLUT (`freeglut`)
- GCC / G++

```bash
# Ubuntu/Debian
sudo apt install freeglut3-dev
```

---

## Compilación

```bash
# Compilar versión CUDA
make

# Compilar versión CPU (para comparativa)
make cpu
```

O manualmente:

```bash
# CUDA
nvcc -O3 particles.cu particles_viewer_opengl.cpp -o particles -lglut -lGLU -lGL

# CPU
g++ -O3 -std=c++17 particles_cpu.cpp particles_viewer_opengl.cpp -o particles_cpu -lglut -lGLU -lGL -lpthread
```

---

## Ejecución

```bash
# Versión CUDA
./particles

# Versión CPU
./particles_cpu
```

---

## Controles

| Tecla | Acción |
|-------|--------|
| `↑ / ↓` | Aumentar / disminuir gravedad |
| `O` | Aumentar rebote (bouncyness) |
| `I` | Disminuir rebote (bouncyness) |
| `R` | Resetear gravedad a 9.8 |
| `ESC` | Salir |

---

## Métricas en pantalla

El HUD muestra en tiempo real:

- **FPS** — frames por segundo
- **memcpy D→H** — tiempo de transferencia GPU→CPU y porcentaje del frame
- **K3 colisiones** — tiempo del kernel más costoso y porcentaje del frame

---

## Estructura del proyecto

```
├── particles.cu                  # Versión CUDA — kernels y bucle principal
├── particles_cpu.cpp             # Versión CPU multihilo — misma lógica con std::thread
├── particles_viewer_opengl.hpp   # Struct de estado del visor y declaración de init
├── particles_viewer_opengl.cpp   # Visor OpenGL con GLUT — callbacks y HUD
└── Makefile
```

---

## Rendimiento

Medido con `cudaEvent` en cada frame:

| Métrica | Valor aproximado |
|---------|-----------------|
| FPS | ~60 |
| K3 colisiones | ~93% del frame |
| cudaMemcpy D→H | ~1.5% del frame |

> El kernel dominante es `resolveCollisions` (K3), cuyo coste es variable según la densidad local de partículas.

---

## Versión CPU

`particles_cpu.cpp` implementa la misma simulación usando `std::thread` para paralelizar en CPU. Permite comparar directamente el rendimiento CPU vs GPU con el mismo número de partículas.

---

## Asignatura

Prácticas de **GPUs y Aceleradores** — Universidad, 3º curso.
