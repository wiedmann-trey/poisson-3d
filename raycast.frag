#version 330 core
in vec3 vPos;
out vec4 fragColor;

uniform sampler3D volumeTex;
uniform vec3 cameraPos;        // Camera position in texture space [0,1]
uniform float stepSize;        // Ray step size (e.g., 0.01)
uniform float densityScale;    // Opacity multiplier (e.g., 2.0)
uniform float volumeMax;
uniform float volumeMin;

vec3 colorMapRainbow(float t) {
    float r = 0.5 + 0.5 * sin(6.283 * (t + 0.0));
    float g = 0.5 + 0.5 * sin(6.283 * (t + 0.33));
    float b = 0.5 + 0.5 * sin(6.283 * (t + 0.66));
    return vec3(r, g, b);
}
void main()
{
    // Ray direction from camera to fragment position
    vec3 rayDir = normalize(vPos - cameraPos);
    
    // Starting point in texture space [0,1]
    vec3 pos = vPos;
    
    // Accumulated color and alpha
    vec4 accum = vec4(0.0);
    
    // Ray marching loop
    for(int i = 0; i < 256; i++) {
        // Check bounds
        if(pos.x < -0.1 || pos.x > 1.1 || 
           pos.y < -0.1 || pos.y > 1.1 || 
           pos.z < -0.1 || pos.z > 1.1) {
            break;
        }
        
        // Sample volume
        float sample = texture(volumeTex, pos).r;
        sample = (sample - volumeMin) / (volumeMax - volumeMin);
        float density = sample/64;
        vec4 sampleColor = vec4(colorMapRainbow(sample), 0);
        
        // Simple transfer function (grayscale)
        sampleColor.a = density * densityScale;
        
        // Front-to-back alpha blending
        sampleColor.rgb *= sampleColor.a;
        accum += (1.0 - accum.a) * sampleColor;
        
        // Early ray termination
        if(accum.a > 0.95) break;
        
        // Step along ray
        pos += rayDir * stepSize;
    }
    
    fragColor = accum;
}
