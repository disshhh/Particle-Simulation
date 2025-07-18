#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void main() {
    float size = 0.05;
    vec4 pos = gl_in[0].gl_Position;

    gl_Position = pos + vec4(-size, -size, 0.0, 0.0);
    EmitVertex();

    gl_Position = pos + vec4( size, -size, 0.0, 0.0);
    EmitVertex();

    gl_Position = pos + vec4(-size,  size, 0.0, 0.0);
    EmitVertex();

    gl_Position = pos + vec4( size,  size, 0.0, 0.0);
    EmitVertex();

    EndPrimitive();
}
