 __kernel void update_particles(
    __global float4* positions, 
    __global float4* velocities, 
    const float time_step, 
    const float gravity,
    const float restitution,
    const float radius,
    const int num_particles) {

    int i = get_global_id(0);
    
    // Update velocity based on gravity
    velocities[i].y -= gravity * time_step;
    

    // Update position based on velocity
    positions[i].x += velocities[i].x * time_step;
    positions[i].y += velocities[i].y * time_step;
    positions[i].z += velocities[i].z * time_step;

    // bounding box //
    if (positions[i].y <= 0 && velocities[i].y < 0) {
	positions[i].y = 0;
	velocities[i].y *= -restitution;
    }
    if (positions[i].x < 0) {
        positions[i].x = 0;
        velocities[i].x *= -restitution;
    } else if (positions[i].x > 2) {
        positions[i].x = 2;
        velocities[i].x *= -restitution;
    }

    // Collision detection with other particles
    for (int j = 0; j < num_particles; j++) {
        if (i != j) {
            float dx = positions[i].x - positions[j].x;
            float dy = positions[i].y - positions[j].y;
            float dz = positions[i].z - positions[j].z;
            float distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (distance < 2 * radius) {
                // Simple elastic collision response
                float nx = dx / distance;
                float ny = dy / distance;
                float nz = dz / distance;

                float rel_vel_x = velocities[i].x - velocities[j].x;
                float rel_vel_y = velocities[i].y - velocities[j].y;
                float rel_vel_z = velocities[i].z - velocities[j].z;

                float dot_product = rel_vel_x * nx + rel_vel_y * ny + rel_vel_z * nz;

                velocities[i].x -= dot_product * nx;
                velocities[i].y -= dot_product * ny;
                velocities[i].z -= dot_product * nz;

                velocities[j].x += dot_product * nx;
                velocities[j].y += dot_product * ny;
                velocities[j].z += dot_product * nz;
            }
        }
    }
}

