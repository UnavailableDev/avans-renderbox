float2 index2pos(
   int index,
   const int width,
   const int height
){
   return (float2){index % width, index / width};
}

int pos2index(
   float2 pos,
   const int width
) {
   return (pos[0] + pos[1] * width);
}

// float divergence(float2 vec){


//    return 0.0;
// }

// float2 dirivative(float2 vec, float width, float height, int num_cells){
//    aaa = vec[0] * (width)
// }

// __kernel void update_temperature(    
//    // __global float4* positions, 
//    __global float4* velocities, 
//    __global float4* velocities, 
//    const float delta_t, 
//    //  const float gravity,
//    const float restitution,
//    const float viscosity,
//    //  const float radius,
//    const int num_particles,
//    const int width,
//    const int height) {
   
// }



// __kernel void updateNavier(
//       __global float4* velocities,
//       const float delta_t,
//       ) {

//    // delta_vel/delta_t = -(directional_derivative)*velocities + v*divergence^2 - 1/d*divergence*pressure + force;
//    // delta_vel = delta_t * ( advection + Diffusion - Pressure );
//    // delta_vel = delta_t * ( advection + Diffusion - Pressure + force );

//    // Simplest form
//    // delta_vel = delta_t * ( advection + force );


//    // mass conservation condition:
//    divergence = 0
// }

// float clamp(
//    float min,
//    float val,
//    float max
// ) {
//    // min 0
//    // val index
//    // max width*height

// }

__kernel void update_velocity(
   __global float* vel_u,
   __global float* vel_v,
   __global float* cell_flow,
   const float delta_t,
   const float2 force
) {
   int i = get_global_id(0);

   if (cell_flow[i] <= 0.0f){
      vel_u[i] = 0;
      vel_v[i] = 0;
      return;
   }


   vel_u[i] += delta_t * force[0];
   vel_v[i] += delta_t * force[1];
}

__kernel void update_divergence(
   __global float* u,
   __global float* v,
   const float delta_t,
   const int width,
   const int height,
   __global float* cell_flow,
   const float overrelaxation
) {
   int i = get_global_id(0);

   if (cell_flow[i] <= 0.0f)
      return;

   // Bottom left is defined as origin point in coordinate system
   // cannot be out of range when range(0-num_cells)

   float divergence = overrelaxation * (u[i] + v[i] - u[i+1] - v[i+1]);
   // float divergence = overrelaxation * (-u[i] - v[i] + u[i+1] + v[i+1]);

   float valid[4];
   if (i - 1 < 0){
      //left wall
      valid[0] = 0.0f;
      valid[1] = 0.0f;
   } else if (i - height < 0){
      //floor
      valid[0] = 0.0f;
      valid[1] = cell_flow[i-1];
   } else {
      // Normal condition
      valid[0] = cell_flow[i-height];
      valid[1] = cell_flow[i-1];
   } 
   if (i + 1 > width * height){
      //right wall
      valid[2] = 0.0f;
      valid[3] = 0.0f;
   } else if (i + height > width * height){
      //ceiling
      valid[2] = 0.0f;
      valid[3] = cell_flow[i+1];
   } else {
      // Normal condition
      valid[2] = cell_flow[i+height];
      valid[3] = cell_flow[i+1];
   }
   float cells_density = valid[0] + valid[1] + valid[2] + valid[3];
   u[i] -= (divergence)/4; // [i-height]
   u[i+1] += (divergence)/4; // [+height]
   v[i] -= (divergence)/4; // [i-1]
   v[i+1] += (divergence)/4; // [i+1]

   // u[i] -= (divergence * valid[1])/cells_density; // [i-height]
   // v[i] -= (divergence * valid[0])/cells_density; // [i-1]
   // u[i+1] += (divergence * valid[3])/cells_density; // [+height]
   // v[i+1] += (divergence * valid[2])/cells_density; // [i+1]
   // float cells_density = cell_flow[i-1] + cell_flow[i+1] + cell_flow[i-height] + cell_flow[i+height];
   // u[i] += (divergence * cell_flow[i-height])/cells_density; // [i-height]
   // v[i] += (divergence * cell_flow[i-1])/cells_density; // [i-1]
   // u[i] += (divergence * cell_flow[i+height])/cells_density; // [+height]
   // v[i] += (divergence * cell_flow[i+1])/cells_density; // [i+1]
}

__kernel void update_pressure() {
   
}

__kernel void update_viscosity() {
   
}

__kernel void update_advection(
   __global float* u,
   __global float* v,
   const float delta_t,
   const int width,
   const int height
) {
   // where did it come from?
   int i = get_global_id(0);

   float2 curr_vec = {u[i] + u[i+1], v[i] + v[i+1]};
   float2 curr_pos = index2pos(i, width, height);
   float2 prev_pos = curr_pos - (delta_t * curr_vec);

   float2 prev_vec;
   int index2 = pos2index(prev_pos, width);
   u[i] = (u[index2] + u[index2+1] + u[index2+height] + u[index2+height+1])/4;
   v[i] = (v[index2] + v[index2+1] + v[index2+height] + v[index2+height+1])/4;

}

