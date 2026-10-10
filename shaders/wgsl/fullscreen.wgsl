// A fullscreen triangle from the vertex index alone (no vertex buffer): vertices
// (-1,-1), (3,-1), (-1,3) cover the whole viewport. Draw with 3 vertices.

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
