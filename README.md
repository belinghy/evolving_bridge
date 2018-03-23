# 2D Evolving Bridge

![screenshot](https://img.youtube.com/vi/NDPkao_OBoA/0.jpg)

[Video Demo](https://www.youtube.com/watch?v=NDPkao_OBoA)

### Description
The four purple arrows in the video are static loads on the bridge; the loads are equal in magnitude indicated by the length of the arrows. The bridge is anchored at the bottom left and bottom right, where the green triangles are. Solid beams are in compression, dotted beams are in tension. Blue means the beam is holding up, and red beams are about to fail. The final evolution contains no red beams, meaning that it's supporting the load.

Due to symmetry of the problem, we expect the optimal design to be symmetrical. The algorithm spends a lot of time near the end fine-tuning the design, but still cannot achieve symmetry. It seems a more sophisticated mutation algorithm is needed for faster convergence, and perhaps symmetry needs to be explicitly specified.
