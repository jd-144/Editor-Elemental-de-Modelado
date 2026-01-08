#version 330 core

in vec3 fragPos;
in vec3 fragNormal;
in vec3 fragColor;

out vec4 outColor;

uniform vec3 lightPos = vec3(2.0, 2.0, 2.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 viewPos = vec3(0.0, 0.0, 0.0);
uniform int drawEdges = 0;

void main()
{
    if (drawEdges == 1) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Verificar y normalizar la normal
    vec3 normal = fragNormal;
    
    // Cálculos de iluminación
    vec3 lightDir = normalize(lightPos - fragPos);
    vec3 viewDir = normalize(viewPos - fragPos);
    
    // Ambient
    vec3 ambient = 0.3 * lightColor;
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normalize(normal), halfwayDir), 0.0), 32.0);
    vec3 specular = 0.2 * spec * lightColor;
    
    // Resultado final
    vec3 result = (ambient + diffuse + specular) * fragColor;
    outColor = vec4(result, 1.0);
}