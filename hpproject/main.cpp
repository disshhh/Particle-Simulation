#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
const int MAX_PARTICLES = 5000;  

// Camera
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Physics constants
const float GRAVITY = 0.2f;
const float FRICTION = 0.99f;
const float RESTITUTION = 0.8f;
const float COLLISION_THRESHOLD = 0.05f;
const float SHOCKWAVE_RADIUS = 1.8f;

// Boundary dimensions
const float BOUNDARY_X = 5.0f;
const float BOUNDARY_Y = 3.0f;
const float BOUNDARY_Z = 5.0f;

// Particle structure for CPU-side data
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 color;
    float size;
    float life;
    float mass;
    bool active;
    bool fixed;  // Fixed particles don't move (like walls)
};

// For SSBO (Shader Storage Buffer Object) data structure
struct GPUParticle {
    float posX, posY, posZ;
    float velX, velY, velZ;
    float colorR, colorG, colorB;
    float size;
    float life;
    float mass;
    int active;
    int fixed;
};

std::vector<Particle> particles;
unsigned int VAO, VBO[2], shaderProgram;
unsigned int boundaryVAO, boundaryVBO, boundaryShader;

// GPU compute resources
unsigned int computeShader, particleSSBO;
bool useGPUCompute = true;  // Flag to toggle GPU computing

// Shader sources
const char* vertexShaderSource = R"(
#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
out vec3 Color;
uniform mat4 projection;
uniform mat4 view;
void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    Color = aColor;
}
)";

const char* geometryShaderSource = R"(
#version 430 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in vec3 Color[];
out vec3 FragColor;
out vec2 TexCoord;
uniform mat4 view;
uniform float pointSize;

void main() {
    vec4 center = gl_in[0].gl_Position;
    float size = pointSize;
    vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 up = vec3(view[0][1], view[1][1], view[2][1]);
    FragColor = Color[0];

    TexCoord = vec2(0.0, 0.0);
    gl_Position = center + vec4(-right - up, 0.0) * size;
    EmitVertex();

    TexCoord = vec2(1.0, 0.0);
    gl_Position = center + vec4(right - up, 0.0) * size;
    EmitVertex();

    TexCoord = vec2(0.0, 1.0);
    gl_Position = center + vec4(-right + up, 0.0) * size;
    EmitVertex();

    TexCoord = vec2(1.0, 1.0);
    gl_Position = center + vec4(right + up, 0.0) * size;
    EmitVertex();

    EndPrimitive();
}
)";

const char* fragmentShaderSource = R"(
#version 430 core
in vec3 FragColor;
in vec2 TexCoord;
out vec4 FragColorOut;
void main() {
    vec2 coord = TexCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float alpha = smoothstep(0.5, 0.3, dist);  // Softer edge for realistic look
    FragColorOut = vec4(FragColor, alpha);
}
)";

// Compute shader for GPU physics simulation
const char* computeShaderSource = R"(
#version 430 core
layout(local_size_x = 256) in; // WorkGroup size

// Define the same structure as in C++ code
struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 color;
    float size;
    float life;
    float mass;
    int active;
    int fixed;
};

// Bind the SSBO to binding point 0
layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// Physics constants
uniform float deltaTime;
uniform float gravity;
uniform float friction;
uniform float restitution;
uniform float boundaryX;
uniform float boundaryY;
uniform float boundaryZ;

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    // Check bounds
    if (index >= particles.length() || particles[index].active == 0)
        return;
        
    // Skip fixed particles
    if (particles[index].fixed == 1)
        return;
        
    // Apply gravity
    particles[index].velocity.y -= gravity * deltaTime;
    
    // Apply friction
    particles[index].velocity *= pow(friction, deltaTime * 60.0);
    
    // Update position
    particles[index].position += particles[index].velocity * deltaTime * 60.0;
    
    // Boundary collision
    if (particles[index].position.x < -boundaryX) {
        particles[index].position.x = -boundaryX;
        particles[index].velocity.x = -particles[index].velocity.x * restitution;
    }
    if (particles[index].position.x > boundaryX) {
        particles[index].position.x = boundaryX;
        particles[index].velocity.x = -particles[index].velocity.x * restitution;
    }
    if (particles[index].position.y < -boundaryY) {
        particles[index].position.y = -boundaryY;
        particles[index].velocity.y = -particles[index].velocity.y * restitution;
    }
    if (particles[index].position.y > boundaryY) {
        particles[index].position.y = boundaryY;
        particles[index].velocity.y = -particles[index].velocity.y * restitution;
    }
    if (particles[index].position.z < -boundaryZ) {
        particles[index].position.z = -boundaryZ;
        particles[index].velocity.z = -particles[index].velocity.z * restitution;
    }
    if (particles[index].position.z > boundaryZ) {
        particles[index].position.z = boundaryZ;
        particles[index].velocity.z = -particles[index].velocity.z * restitution;
    }
    
    // Decrease life
    particles[index].life -= deltaTime;
    if (particles[index].life <= 0.0)
        particles[index].active = 0;
}
)";

// Addition: Compute shader for particle collision detection and resolution
const char* collisionComputeShaderSource = R"(
#version 430 core
layout(local_size_x = 256) in; // WorkGroup size

struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 color;
    float size;
    float life;
    float mass;
    int active;
    int fixed;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform int particleCount;
uniform float collisionThreshold;
uniform float restitution;

shared Particle sharedParticles[256]; // Shared memory for local workgroup

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint localId = gl_LocalInvocationID.x;
    
    // Load current particle to shared memory
    Particle currentParticle;
    if (index < particleCount) {
        currentParticle = particles[index];
    }
    
    // Early exit if this particle isn't active
    if (index >= particleCount || currentParticle.active == 0 || currentParticle.fixed == 1) {
        return;
    }
    
    // Memory barrier to ensure all threads have loaded into shared memory
    barrier();
    
    // Process a subset of collisions
    // We divide the work by having each thread check against particles that come after it
    for (int j = 0; j < particleCount; j++) {
        if (j == index || particles[j].active == 0) continue;
        
        // Compute distance
        vec3 diff = particles[j].position - currentParticle.position;
        float distance = length(diff);
        float collisionDist = currentParticle.size + particles[j].size + collisionThreshold;
        
        // Only process collisions
        if (distance < collisionDist && distance > 0.0001) {
            vec3 normal = normalize(diff);
            
            // If other particle is fixed, handle one-sided collision
            if (particles[j].fixed == 1) {
                float depth = collisionDist - distance;
                
                // Position correction
                particles[index].position -= normal * depth;
                
                // Velocity correction
                float projVel = dot(particles[index].velocity, normal);
                if (projVel < 0) { // Moving towards the fixed particle
                    particles[index].velocity -= (1.0 + restitution) * projVel * normal;
                }
            }
            // Otherwise handle two-sided collision
            else {
                float totalMass = currentParticle.mass + particles[j].mass;
                float ratio1 = currentParticle.mass / totalMass;
                float ratio2 = particles[j].mass / totalMass;
                
                // Position correction (avoid race conditions by only adjusting this particle)
                float depth = (collisionDist - distance) * 0.5;
                atomicAdd(particles[index].position.x, -normal.x * depth);
                atomicAdd(particles[index].position.y, -normal.y * depth);
                atomicAdd(particles[index].position.z, -normal.z * depth);
                
                // Velocity correction
                vec3 relativeVel = particles[j].velocity - currentParticle.velocity;
                float velAlongNormal = dot(relativeVel, normal);
                
                if (velAlongNormal < 0) {
                    float j = -(1.0 + restitution) * velAlongNormal;
                    j /= (1.0 / currentParticle.mass + 1.0 / particles[j].mass);
                    
                    vec3 impulse = j * normal * 1.2; // Amplified impulse
                    
                    // Apply impulse to this particle only
                    // Other particle will handle its own collision in its thread
                    atomicAdd(particles[index].velocity.x, -impulse.x / currentParticle.mass);
                    atomicAdd(particles[index].velocity.y, -impulse.y / currentParticle.mass);
                    atomicAdd(particles[index].velocity.z, -impulse.z / currentParticle.mass);
                }
            }
        }
    }
}
)";

// Boundary shaders
const char* boundaryVertexShaderSource = R"(
#version 430 core
layout(location = 0) in vec3 aPos;
uniform mat4 projection;
uniform mat4 view;
void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
}
)";

const char* boundaryFragmentShaderSource = R"(
#version 430 core
out vec4 FragColor;
uniform vec3 boundaryColor;
void main() {
    FragColor = vec4(boundaryColor, 0.3);  // Semi-transparent boundaries
}
)";

// Addition: Compute shader for explosive effects
const char* explosionShaderSource = R"(
#version 430 core
layout(local_size_x = 256) in;

struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 color;
    float size;
    float life;
    float mass;
    int active;
    int fixed;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// Explosion parameters
uniform vec3 explosionCenter;
uniform float explosionRadius;
uniform float explosionForce;
uniform int particleCount;

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (index >= particleCount || particles[index].active == 0 || particles[index].fixed == 1)
        return;
        
    // Calculate distance to explosion
    vec3 direction = particles[index].position - explosionCenter;
    float distance = length(direction);
    
    // Apply force if within radius
    if (distance < explosionRadius) {
        float forceFactor = (explosionRadius - distance) / explosionRadius; // Linear falloff
        forceFactor = forceFactor * forceFactor; // Quadratic falloff
        
        // Normalize direction and apply force
        vec3 forceDir = normalize(direction);
        vec3 force = forceDir * forceFactor * explosionForce;
        
        // Add to particle velocity
        particles[index].velocity += force / particles[index].mass;
        
        // Colorize particle based on explosion proximity
        particles[index].color = mix(vec3(1.0, 0.5, 0.0), particles[index].color, distance/explosionRadius);
    }
}
)";

unsigned int collisionComputeShader, explosionComputeShader;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

unsigned int compileShader(GLenum type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "Shader compilation error:\n" << infoLog << std::endl;
    }
    return shader;
}

void initComputeShaders() {
    // Create and compile compute shader
    computeShader = compileShader(GL_COMPUTE_SHADER, computeShaderSource);
    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    glDeleteShader(computeShader);
    computeShader = computeProgram;

    // Create and compile collision compute shader
    collisionComputeShader = compileShader(GL_COMPUTE_SHADER, collisionComputeShaderSource);
    GLuint collisionProgram = glCreateProgram();
    glAttachShader(collisionProgram, collisionComputeShader);
    glLinkProgram(collisionProgram);
    glDeleteShader(collisionComputeShader);
    collisionComputeShader = collisionProgram;

    // Create and compile explosion compute shader
    explosionComputeShader = compileShader(GL_COMPUTE_SHADER, explosionShaderSource);
    GLuint explosionProgram = glCreateProgram();
    glAttachShader(explosionProgram, explosionComputeShader);
    glLinkProgram(explosionProgram);
    glDeleteShader(explosionComputeShader);
    explosionComputeShader = explosionProgram;

    // Create SSBO for particles
    glGenBuffers(1, &particleSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_PARTICLES * sizeof(GPUParticle), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    std::cout << "GPU Compute shaders initialized." << std::endl;
}

void spawnParticle(glm::vec3 position, glm::vec3 velocity, glm::vec3 color, float size, float life, float mass = 1.0f, bool fixed = false) {
    if (particles.size() < MAX_PARTICLES) {
        particles.push_back({ position, velocity, color, size, life, mass, true, fixed });
    }
}

void explode(glm::vec3 origin, glm::vec3 baseVelocity, int count, glm::vec3 color, float totalMass) {
    float fragmentMass = totalMass / count;

    // Ensure we don't spawn too many particles
    count = std::min(count, 30 + static_cast<int>(totalMass * 5));

    for (int i = 0; i < count; ++i) {
        // Check if we've reached the maximum particle count
        if (particles.size() >= MAX_PARTICLES) {
            break;
        }

        // Create realistic explosion pattern
        glm::vec3 direction = glm::sphericalRand(1.0f);
        float speed = glm::linearRand(0.5f, 1.5f);
        glm::vec3 fragVelocity = baseVelocity + direction * speed;

        // Vary size and life based on mass and randomness
        float fragSize = 0.03f + 0.05f * (fragmentMass / 0.1f) * glm::linearRand(0.8f, 1.2f);
        float fragLife = 4.0f + glm::linearRand(0.0f, 2.0f);

        // Vary color slightly
        glm::vec3 fragColor = color * glm::vec3(
            glm::linearRand(0.9f, 1.1f),
            glm::linearRand(0.9f, 1.1f),
            glm::linearRand(0.9f, 1.1f)
        );

        spawnParticle(origin, fragVelocity, fragColor, fragSize, fragLife, fragmentMass);
    }
}

// GPU-based explosion using compute shader
void explodeGPU(glm::vec3 origin, float radius, float force) {
    glUseProgram(explosionComputeShader);

    // Set explosion parameters
    glUniform3f(glGetUniformLocation(explosionComputeShader, "explosionCenter"),
        origin.x, origin.y, origin.z);
    glUniform1f(glGetUniformLocation(explosionComputeShader, "explosionRadius"), radius);
    glUniform1f(glGetUniformLocation(explosionComputeShader, "explosionForce"), force);
    glUniform1i(glGetUniformLocation(explosionComputeShader, "particleCount"), particles.size());

    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleSSBO);

    // Dispatch compute shader
    int numGroups = (particles.size() + 255) / 256;
    glDispatchCompute(numGroups, 1, 1);

    // Memory barrier to ensure compute shader writes are visible
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Function to update particle data from CPU to GPU SSBO
void updateParticleSSBO() {
    std::vector<GPUParticle> gpuParticles(particles.size());

    for (size_t i = 0; i < particles.size(); i++) {
        gpuParticles[i].posX = particles[i].position.x;
        gpuParticles[i].posY = particles[i].position.y;
        gpuParticles[i].posZ = particles[i].position.z;
        gpuParticles[i].velX = particles[i].velocity.x;
        gpuParticles[i].velY = particles[i].velocity.y;
        gpuParticles[i].velZ = particles[i].velocity.z;
        gpuParticles[i].colorR = particles[i].color.r;
        gpuParticles[i].colorG = particles[i].color.g;
        gpuParticles[i].colorB = particles[i].color.b;
        gpuParticles[i].size = particles[i].size;
        gpuParticles[i].life = particles[i].life;
        gpuParticles[i].mass = particles[i].mass;
        gpuParticles[i].active = particles[i].active ? 1 : 0;
        gpuParticles[i].fixed = particles[i].fixed ? 1 : 0;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particles.size() * sizeof(GPUParticle), gpuParticles.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// Function to read back data from GPU SSBO to CPU particles
void readbackParticleSSBO() {
    std::vector<GPUParticle> gpuParticles(particles.size());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleSSBO);
    void* ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    memcpy(gpuParticles.data(), ptr, particles.size() * sizeof(GPUParticle));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for (size_t i = 0; i < particles.size(); i++) {
        particles[i].position = glm::vec3(gpuParticles[i].posX, gpuParticles[i].posY, gpuParticles[i].posZ);
        particles[i].velocity = glm::vec3(gpuParticles[i].velX, gpuParticles[i].velY, gpuParticles[i].velZ);
        particles[i].color = glm::vec3(gpuParticles[i].colorR, gpuParticles[i].colorG, gpuParticles[i].colorB);
        particles[i].size = gpuParticles[i].size;
        particles[i].life = gpuParticles[i].life;
        particles[i].mass = gpuParticles[i].mass;
        particles[i].active = gpuParticles[i].active != 0;
        particles[i].fixed = gpuParticles[i].fixed != 0;
    }
}

// GPU-accelerated physics update
void updateParticlesGPU(float dt) {
    // Ensure dt is not zero
    if (dt < 0.001f) dt = 0.001f;

    // Cap dt to prevent simulation instability
    dt = std::min(dt, 0.05f);

    // Update particle SSBO with current CPU data
    updateParticleSSBO();

    // Physics Update Compute Shader
    glUseProgram(computeShader);

    // Set uniforms
    glUniform1f(glGetUniformLocation(computeShader, "deltaTime"), dt);
    glUniform1f(glGetUniformLocation(computeShader, "gravity"), GRAVITY);
    glUniform1f(glGetUniformLocation(computeShader, "friction"), FRICTION);
    glUniform1f(glGetUniformLocation(computeShader, "restitution"), RESTITUTION);
    glUniform1f(glGetUniformLocation(computeShader, "boundaryX"), BOUNDARY_X);
    glUniform1f(glGetUniformLocation(computeShader, "boundaryY"), BOUNDARY_Y);
    glUniform1f(glGetUniformLocation(computeShader, "boundaryZ"), BOUNDARY_Z);

    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleSSBO);

    // Dispatch compute shader
    int numGroups = (particles.size() + 255) / 256;
    glDispatchCompute(numGroups, 1, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Collision Detection & Resolution Compute Shader
    glUseProgram(collisionComputeShader);

    // Set uniforms
    glUniform1i(glGetUniformLocation(collisionComputeShader, "particleCount"), particles.size());
    glUniform1f(glGetUniformLocation(collisionComputeShader, "collisionThreshold"), COLLISION_THRESHOLD);
    glUniform1f(glGetUniformLocation(collisionComputeShader, "restitution"), RESTITUTION);

    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particleSSBO);

    // Dispatch compute shader
    glDispatchCompute(numGroups, 1, 1);

    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Read back results to CPU
    readbackParticleSSBO();

    // Clean up particles that have expired
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return !p.active; }),
        particles.end());
}

// CPU-based physics update with OpenMP parallelization
void updateParticlesCPU(float dt) {
    // Ensure dt is not zero
    if (dt < 0.001f) dt = 0.001f;

    // Cap dt to prevent simulation instability
    dt = std::min(dt, 0.05f);

    // Print simulation stats occasionally
    static float printTimer = 0.0f;
    printTimer += dt;
    if (printTimer > 1.0f) {
        std::cout << "Active particles: " << particles.size() << ", dt: " << dt << std::endl;
        printTimer = 0.0f;
    }

    // Apply physics to all particles in parallel using OpenMP
#pragma omp parallel for
    for (int i = 0; i < particles.size(); i++) {
        auto& p = particles[i];
        if (!p.active) continue;

        // Skip fixed particles
        if (p.fixed) continue;

        // Apply gravity
        p.velocity.y -= GRAVITY * dt;

        // Apply friction (reduced friction for better movement)
        p.velocity *= pow(FRICTION, dt * 60.0f);  // Scale friction by framerate

        // Update position
        p.position += p.velocity * dt * 60.0f;  // Scale velocity by framerate

        // Boundary collision with walls
        if (p.position.x < -BOUNDARY_X) {
            p.position.x = -BOUNDARY_X;
            p.velocity.x = -p.velocity.x * RESTITUTION;
        }
        if (p.position.x > BOUNDARY_X) {
            p.position.x = BOUNDARY_X;
            p.velocity.x = -p.velocity.x * RESTITUTION;
        }
        if (p.position.y < -BOUNDARY_Y) {
            p.position.y = -BOUNDARY_Y;
            p.velocity.y = -p.velocity.y * RESTITUTION;
        }
        if (p.position.y > BOUNDARY_Y) {
            p.position.y = BOUNDARY_Y;
            p.velocity.y = -p.velocity.y * RESTITUTION;
        }
        if (p.position.z < -BOUNDARY_Z) {
            p.position.z = -BOUNDARY_Z;
            p.velocity.z = -p.velocity.z * RESTITUTION;
        }
        if (p.position.z > BOUNDARY_Z) {
            p.position.z = BOUNDARY_Z;
            p.velocity.z = -p.velocity.z * RESTITUTION;
        }

        // Decrease life
        p.life -= dt;
        if (p.life <= 0.0f) p.active = false;
    }

    // Collect pairs of particles that might collide
    std::vector<std::pair<int, int>> collisionPairs;

    // Use OpenMP to parallelize collision detection
#pragma omp parallel
    {
        // Each thread will have its own private collection of collision pairs
        std::vector<std::pair<int, int>> privateCollisionPairs;

        // Divide work among threads
#pragma omp for schedule(dynamic, 50)
        for (int i = 0; i < particles.size(); ++i) {
            if (!particles[i].active) continue;

            // Check a subset of particles for better performance
            for (int j = i + 1; j < particles.size() && j < i + 100; ++j) {
                if (!particles[j].active) continue;

                // Compute distance between particles
                float dist = glm::distance(particles[i].position, particles[j].position);
                float collisionRadius = (particles[i].size + particles[j].size);

                // If particles are close enough, add to collision pairs
                if (dist < collisionRadius + COLLISION_THRESHOLD) {
                    privateCollisionPairs.emplace_back(i, j);
                }
            }
        }

        // Merge thread-local results into global collection// Merge thread-local results into global collection
#pragma omp critical
        {
            collisionPairs.insert(collisionPairs.end(),
                privateCollisionPairs.begin(),
                privateCollisionPairs.end());
        }
    } // End OpenMP parallel region

    // Resolve collisions
    for (const auto& pair : collisionPairs) {
        int i = pair.first;
        int j = pair.second;

        if (!particles[i].active || !particles[j].active) continue;

        glm::vec3 diff = particles[j].position - particles[i].position;
        float dist = glm::length(diff);

        // Skip if distance is zero (would cause division by zero)
        if (dist < 0.0001f) continue;

        // Calculate overlap
        float collisionDist = particles[i].size + particles[j].size + COLLISION_THRESHOLD;
        float overlap = collisionDist - dist;

        if (overlap > 0) {
            glm::vec3 normal = diff / dist; // Normalize the difference vector

            // If one particle is fixed, handle one-sided collision
            if (particles[i].fixed && !particles[j].fixed) {
                // Move the non-fixed particle away
                particles[j].position += normal * overlap;

                // Reflect velocity if moving towards fixed particle
                float projVel = glm::dot(particles[j].velocity, normal);
                if (projVel < 0) {
                    particles[j].velocity -= (1.0f + RESTITUTION) * projVel * normal;
                }
            }
            else if (!particles[i].fixed && particles[j].fixed) {
                // Move the non-fixed particle away
                particles[i].position -= normal * overlap;

                // Reflect velocity if moving towards fixed particle
                float projVel = glm::dot(particles[i].velocity, normal);
                if (projVel > 0) {
                    particles[i].velocity -= (1.0f + RESTITUTION) * projVel * normal;
                }
            }
            // If both particles are movable
            else if (!particles[i].fixed && !particles[j].fixed) {
                // Calculate mass ratio for position resolution
                float totalMass = particles[i].mass + particles[j].mass;
                float ratio1 = particles[i].mass / totalMass;
                float ratio2 = particles[j].mass / totalMass;

                // Move particles apart based on mass ratio
                particles[i].position -= normal * overlap * ratio2;
                particles[j].position += normal * overlap * ratio1;

                // Calculate impulse for velocity change
                glm::vec3 relativeVel = particles[j].velocity - particles[i].velocity;
                float velAlongNormal = glm::dot(relativeVel, normal);

                // Only resolve if particles are moving toward each other
                if (velAlongNormal < 0) {
                    float j = -(1.0f + RESTITUTION) * velAlongNormal;
                    j /= (1.0f / particles[i].mass + 1.0f / particles[j].mass);

                    glm::vec3 impulse = j * normal;

                    // Apply impulse based on mass
                    particles[i].velocity -= impulse / particles[i].mass;
                    particles[j].velocity += impulse / particles[j].mass;
                }
            }
        }
    }

    // Remove inactive particles
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return !p.active; }),
        particles.end());
}

// Function to update particles (switches between CPU and GPU modes)
void updateParticles(float dt) {
    if (useGPUCompute) {
        updateParticlesGPU(dt);
    }
    else {
        updateParticlesCPU(dt);
    }
}

void setupShaders() {
    // Vertex shader
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);

    // Geometry shader
    unsigned int geometryShader = compileShader(GL_GEOMETRY_SHADER, geometryShaderSource);

    // Fragment shader
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Shader program linking error:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    glDeleteShader(fragmentShader);

    // Setup boundary shaders
    vertexShader = compileShader(GL_VERTEX_SHADER, boundaryVertexShaderSource);
    fragmentShader = compileShader(GL_FRAGMENT_SHADER, boundaryFragmentShaderSource);

    boundaryShader = glCreateProgram();
    glAttachShader(boundaryShader, vertexShader);
    glAttachShader(boundaryShader, fragmentShader);
    glLinkProgram(boundaryShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void setupBoundary() {
    // Vertices for a wireframe cube
    float boundaryVertices[] = {
        // Bottom face
        -BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,

        // Top face
        -BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,

        // Vertical edges
        -BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,
        -BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, -BOUNDARY_Z,
        BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, -BOUNDARY_Y, BOUNDARY_Z,
        -BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z
    };

    // Create VAO and VBO for boundary
    glGenVertexArrays(1, &boundaryVAO);
    glGenBuffers(1, &boundaryVBO);

    glBindVertexArray(boundaryVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boundaryVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(boundaryVertices), boundaryVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void initializeScene() {
    // Add walls by creating fixed particles along boundaries
    float wallParticleSize = 0.1f;
    int wallDensity = 5; // Adjust based on performance

    // Create a grid of fixed particles for each wall
    for (int x = -wallDensity; x <= wallDensity; ++x) {
        for (int z = -wallDensity; z <= wallDensity; ++z) {
            // Bottom wall
            spawnParticle(
                glm::vec3(x * (BOUNDARY_X * 2 / wallDensity), -BOUNDARY_Y, z * (BOUNDARY_Z * 2 / wallDensity)),
                glm::vec3(0.0f), glm::vec3(0.3f, 0.3f, 0.8f), wallParticleSize, 9999.0f, 99999.0f, true
            );

            // Optional: Top wall
            // spawnParticle(
            //     glm::vec3(x * (BOUNDARY_X * 2 / wallDensity), BOUNDARY_Y, z * (BOUNDARY_Z * 2 / wallDensity)),
            //     glm::vec3(0.0f), glm::vec3(0.3f, 0.3f, 0.8f), wallParticleSize, 9999.0f, 99999.0f, true
            // );
        }
    }

    // Create some normal particles
    for (int i = 0; i < 500; ++i) {
        float size = glm::linearRand(0.05f, 0.15f);
        float mass = size * size * 10.0f; // Mass proportional to volume

        glm::vec3 pos = glm::vec3(
            glm::linearRand(-BOUNDARY_X + 0.2f, BOUNDARY_X - 0.2f),
            glm::linearRand(-BOUNDARY_Y + 0.2f, BOUNDARY_Y - 0.2f),
            glm::linearRand(-BOUNDARY_Z + 0.2f, BOUNDARY_Z - 0.2f)
        );

        glm::vec3 vel = glm::vec3(
            glm::linearRand(-0.1f, 0.1f),
            glm::linearRand(-0.1f, 0.1f),
            glm::linearRand(-0.1f, 0.1f)
        );

        glm::vec3 color = glm::vec3(
            glm::linearRand(0.3f, 1.0f),
            glm::linearRand(0.3f, 1.0f),
            glm::linearRand(0.3f, 1.0f)
        );

        spawnParticle(pos, vel, color, size, 15.0f, mass);
    }
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Camera movement
    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraUp * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        cameraPos -= cameraUp * cameraSpeed;

    // Toggle GPU simulation with G key
    static bool gKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && !gKeyPressed) {
        useGPUCompute = !useGPUCompute;
        std::cout << "Switched to " << (useGPUCompute ? "GPU" : "CPU") << " computation." << std::endl;
        gKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_RELEASE) {
        gKeyPressed = false;
    }

    // Spawn particles with left mouse button
    static bool mouseLeftPressed = false;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !mouseLeftPressed) {
        // Get current mouse position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Convert screen coordinates to NDC
        float x = (2.0f * xpos) / SCR_WIDTH - 1.0f;
        float y = 1.0f - (2.0f * ypos) / SCR_HEIGHT;

        // Create ray from camera through clicked point
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glm::mat4 invVP = glm::inverse(projection * view);
        glm::vec4 rayClip = glm::vec4(x, y, -1.0, 1.0);
        glm::vec4 rayEye = glm::inverse(projection) * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0, 0.0);
        glm::vec4 rayWorld = glm::inverse(view) * rayEye;
        glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));

        // Spawn multiple particles along the ray
        for (int i = 0; i < 10; i++) {
            float distance = 1.0f + i * 0.2f;
            glm::vec3 spawnPosition = cameraPos + rayDir * distance;

            // Check if inside boundary
            if (abs(spawnPosition.x) < BOUNDARY_X &&
                abs(spawnPosition.y) < BOUNDARY_Y &&
                abs(spawnPosition.z) < BOUNDARY_Z) {

                float size = glm::linearRand(0.05f, 0.15f);
                float mass = size * size * 10.0f;

                spawnParticle(
                    spawnPosition,
                    rayDir * glm::linearRand(0.1f, 0.3f),
                    glm::vec3(glm::linearRand(0.5f, 1.0f), glm::linearRand(0.0f, 0.5f), glm::linearRand(0.0f, 0.3f)),
                    size,
                    10.0f,
                    mass
                );
            }
        }

        mouseLeftPressed = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        mouseLeftPressed = false;
    }

    // Create explosion with right mouse button
    static bool mouseRightPressed = false;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !mouseRightPressed) {
        // Get current mouse position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Convert screen coordinates to NDC
        float x = (2.0f * xpos) / SCR_WIDTH - 1.0f;
        float y = 1.0f - (2.0f * ypos) / SCR_HEIGHT;

        // Create ray from camera through clicked point
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        glm::mat4 invVP = glm::inverse(projection * view);
        glm::vec4 rayClip = glm::vec4(x, y, -1.0, 1.0);
        glm::vec4 rayEye = glm::inverse(projection) * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0, 0.0);
        glm::vec4 rayWorld = glm::inverse(view) * rayEye;
        glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));

        // Find a point along the ray that's within the boundary
        float distance = 3.0f;
        glm::vec3 explosionPoint = cameraPos + rayDir * distance;

        if (useGPUCompute) {
            // Use GPU-accelerated explosion
            explodeGPU(explosionPoint, SHOCKWAVE_RADIUS, 0.5f);
        }
        else {
            // Use CPU explosion
            // Create explosion with particles
            explode(
                explosionPoint,            // Position
                glm::vec3(0.0f),           // Base velocity
                50,                        // Number of particles
                glm::vec3(1.0f, 0.5f, 0.0f), // Orange/red color
                5.0f                       // Total mass
            );
        }

        mouseRightPressed = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
        mouseRightPressed = false;
    }
}

// Mouse callback for camera control
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Make sure pitch stays within bounds
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    // Update camera front vector
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Particle Simulator", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Setup shaders and VAO/VBO
    setupShaders();

    // Initialize boundary
    setupBoundary();

    // Initialize GPU compute shaders
    initComputeShaders();

    // Vertex Array Object for particles
    glGenVertexArrays(1, &VAO);
    glGenBuffers(2, VBO);

    // Initialize the scene with particles
    initializeScene();

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        processInput(window);

        // Clear buffers
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update particles
        updateParticles(deltaTime);

        // Create matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        // Draw boundary
        glUseProgram(boundaryShader);
        glUniformMatrix4fv(glGetUniformLocation(boundaryShader, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(boundaryShader, "view"), 1, GL_FALSE, &view[0][0]);
        glUniform3f(glGetUniformLocation(boundaryShader, "boundaryColor"), 0.5f, 0.5f, 0.8f);

        glBindVertexArray(boundaryVAO);
        glDrawArrays(GL_LINES, 0, 24); // 24 vertices for the cube lines

        // Draw particles
        if (!particles.empty()) {
            glUseProgram(shaderProgram);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
            glUniform1f(glGetUniformLocation(shaderProgram, "pointSize"), 0.05f);

            std::vector<glm::vec3> positions;
            std::vector<glm::vec3> colors;

            // Collect only active particles
            for (const auto& p : particles) {
                if (p.active) {
                    positions.push_back(p.position);
                    colors.push_back(p.color);
                }
            }

            // Bind and update VBOs
            glBindVertexArray(VAO);

            // Position VBO
            glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
            glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), positions.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
            glEnableVertexAttribArray(0);

            // Color VBO
            glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
            glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), colors.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
            glEnableVertexAttribArray(1);

            // Draw particles
            glDrawArrays(GL_POINTS, 0, positions.size());

            // Unbind VAO
            glBindVertexArray(0);
        }

        // Print system stats
        if (currentFrame - lastFrame > 0) {
            static float statsTimer = 0.0f;
            statsTimer += deltaTime;
            if (statsTimer > 1.0f) {
                std::cout << "FPS: " << 1.0f / deltaTime << " | Particles: " << particles.size() << " | Mode: "
                    << (useGPUCompute ? "GPU" : "CPU") << std::endl;
                statsTimer = 0.0f;
            }
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(2, VBO);
    glDeleteProgram(shaderProgram);

    glDeleteVertexArrays(1, &boundaryVAO);
    glDeleteBuffers(1, &boundaryVBO);
    glDeleteProgram(boundaryShader);

    glDeleteProgram(computeShader);
    glDeleteProgram(collisionComputeShader);
    glDeleteProgram(explosionComputeShader);
    glDeleteBuffers(1, &particleSSBO);

    glfwTerminate();
    return 0;
}