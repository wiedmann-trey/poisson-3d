#version 330 core
in vec3 vPos;
out vec4 fragColor;

uniform sampler3D volumeTexPrev;
uniform sampler3D volumeTexNext;
uniform float t;
uniform vec3 cameraPos;        // Camera position in texture space [0,1]
uniform vec3 backgroundColor;
uniform float stepSize; 
uniform float densityScale;    // Opacity multiplier
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
    
    vec4 accum = vec4(0.0);
    
    for(int i = 0; i < 256; i++) {
        if(pos.x < -0.1 || pos.x > 1.1 || 
           pos.y < -0.1 || pos.y > 1.1 || 
           pos.z < -0.1 || pos.z > 1.1) {
            break;
        }
        
        float sample = mix(texture(volumeTexPrev, pos).r, texture(volumeTexNext, pos).r, t);
        sample = clamp(sample, volumeMin, volumeMax);
        sample = (sample - volumeMin) / (volumeMax - volumeMin);
        float density = sample * densityScale;
        vec4 sampleColor = vec4(colorMapRainbow(sample), 0);
        
        sampleColor.a = density;
        
        sampleColor.rgb *= sampleColor.a;
        accum += (1.0 - accum.a) * sampleColor;
        
        if(accum.a > 0.95) break;
        
        pos += rayDir * stepSize;
    }

    vec3 finalColor = accum.rgb + backgroundColor * (1.0 - accum.a);
    fragColor = vec4(finalColor, 1.0);
}
