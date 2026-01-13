#include "util.h"
#include <corecrt_math_defines.h>
#include <iostream>

// Lee el contenido de un archivo de texto y lo almacena en 'destino'.
// Devuelve true si la lectura fue exitosa, false en caso contrario.
bool ReadFile(const char* fileName, string& destino)
{
	ifstream file(fileName);
	if (file.is_open()) {
		string line;
		while (getline(file, line)) {
			destino.append(line);
			destino.append("\n");
		}
		file.close();
		return true;
	}
	fprintf(stderr, "Error: no se pudo abrir %s\n", fileName);
	return false;
}

// Calcula el cuaternión que rota el vector from hacia el vector to
glm::quat quatFromVectors(const glm::vec3& from, const glm::vec3& to) {
    glm::vec3 crossProduct = glm::cross(from, to);
    float dotProduct = glm::dot(from, to);
    float w = sqrt(glm::length(from) * glm::length(from) * glm::length(to) * glm::length(to)) + dotProduct;
    return glm::normalize(glm::quat(w, crossProduct.x, crossProduct.y, crossProduct.z));
}

glm::vec3 screenToWorld(int x, int y, const glm::mat4& viewProjectionMatrix,
    int screenWidth, int screenHeight) {

	// Conversión a coordenadas normalizadas [-1, 1]
    float nx = (2.0f * x) / screenWidth - 1.0f;
    float ny = 1.0f - (2.0f * y) / screenHeight;

    // Crear puntos en espacio de recorte para near y far planes
    glm::vec4 rayClipNear(nx, ny, -1.0f, 1.0f);  // near plane  w = 1 para punto
    glm::vec4 rayClipFar(nx, ny, 1.0f, 1.0f);    // far plane

    // Transformar a espacio real
    glm::mat4 inverseViewProjection = glm::inverse(viewProjectionMatrix);
    glm::vec4 rayWorldNear = inverseViewProjection * rayClipNear;
    glm::vec4 rayWorldFar = inverseViewProjection * rayClipFar;

    // Homogenizar
    rayWorldNear /= rayWorldNear.w;
    rayWorldFar /= rayWorldFar.w;

    // Calcular dirección del rayo
    glm::vec3 rayDirection = glm::normalize(rayWorldFar - rayWorldNear);

    return rayDirection;
}
// Algoritmo de Möller–Trumbore
// Devuelve el punto de intersección si existe, nullptr si no
glm::vec3* rayIntersectsFace(
    const glm::vec3& rayOrigin, const glm::vec3& rayDirection,
    Model& model, const Face& face, unsigned int& triangleIndex, float& outT)
{
    float epsilon = 1e-6f;
    outT = -1.0f;
    for (unsigned int i = 0; i < face.index.size(); i++) {
        auto tri = face.index[i];
        glm::vec3 triangleV0 = model.getVertex()[tri[0]].pos;
        glm::vec3 triangleV1 = model.getVertex()[tri[1]].pos;
        glm::vec3 triangleV2 = model.getVertex()[tri[2]].pos;
        glm::vec3 edge1 = triangleV1 - triangleV0;
        glm::vec3 edge2 = triangleV2 - triangleV0;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

		// Backface culling: solo intersecciones que miran a la cámara
        if (glm::dot(normal, rayDirection) >= 0.0f) {
            continue;
        }
        glm::vec3 rayCrossEdge2 = glm::cross(rayDirection, edge2);
        float det = glm::dot(edge1, rayCrossEdge2);
		// Rayo paralelo al triángulo
        if (fabs(det) < epsilon) {
            continue;
        }
        float invDet = 1.0f / det;
        glm::vec3 sVec = rayOrigin - triangleV0;
		// Coordenadas baricéntricas u
        float u = invDet * glm::dot(sVec, rayCrossEdge2);
        if (u < 0.0f || u > 1.0f) {
            continue;
        }
        glm::vec3 sCrossEdge1 = glm::cross(sVec, edge1);
		// Coordenadas baricéntricas v
        float v = invDet * glm::dot(rayDirection, sCrossEdge1);
        if (v < 0.0f || (u + v > 1.0f)) {
            continue;
        }
		// Distancia t desde el origen del rayo al punto de intersección
        float t = invDet * glm::dot(edge2, sCrossEdge1);
		// Delante del origen del rayo
        if (t >= epsilon) {
            triangleIndex = i;
            outT = t;
            return new glm::vec3(rayOrigin + rayDirection * t);
        }
    }
    return nullptr;
}

// Verifica si un vértice ya existe en la lista de vértices
bool vertexExists(const std::vector<Vertex>& vertices, const glm::vec3& position, float epsilon) {
    for (const auto& vertex : vertices) {
        float distance = glm::length(vertex.pos - position);
        if (distance < epsilon) {
            return true;
        }
    }
    return false;
}

// Subdivide una cara añadiendo un nuevo vértice en el triángulo especificado de la cara
void subdivideFace(Model& model, Face& face, const glm::vec3& newVertexPos, unsigned int trianglePositition) {
    if (trianglePositition >= face.index.size()) return;
	// Evita añadir vértices duplicados o muy cercanos a los existentes
    if (vertexExists(model.getVertex(), newVertexPos)) return;

    std::array<unsigned int, 3> selectedTriangle = face.index[trianglePositition];
    // selectedTriangle = ABC
    glm::vec3 triangleA = model.getVertex()[selectedTriangle[0]].pos;
    glm::vec3 triangleB = model.getVertex()[selectedTriangle[1]].pos;
    glm::vec3 triangleC = model.getVertex()[selectedTriangle[2]].pos;
	float minDistance = 0.001f; // Distancia mínima al vértice existente
    if (glm::length(newVertexPos - triangleA) < minDistance ||
        glm::length(newVertexPos - triangleB) < minDistance ||
        glm::length(newVertexPos - triangleC) < minDistance) {
        return;
    }
    Vertex newVertex(newVertexPos.x, newVertexPos.y, newVertexPos.z);
    model.addVertex(newVertex);

	// Crea tres nuevas caras subdividiendo el triángulo original
    unsigned int newVertexIndex = model.getVertex().size() - 1;
	
    std::vector<std::array<unsigned int, 3>> newTriangles = {
		{selectedTriangle[0], selectedTriangle[1], newVertexIndex}, // Triángulo AB-N
		{selectedTriangle[1], selectedTriangle[2], newVertexIndex}, // Triángulo BC-N
		{selectedTriangle[2], selectedTriangle[0], newVertexIndex} // Triángulo CA-N
    };
    face.index.erase(face.index.begin() + trianglePositition);
    face.index.insert(face.index.end(), newTriangles.begin(), newTriangles.end());
    calculateNormals(model);
}

// Actualiza los buffers de un modelo tras modificar su geometría
void updateModelMesh(Model& model, MeshBuffers& mesh) {
    std::vector<Vertex> vertex = model.getVertex();
    std::vector<unsigned int> index = model.getAllIndex();

    // Actualizar VBO (vértices)
    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertex.size() * sizeof(Vertex), vertex.data(), GL_DYNAMIC_DRAW);
    // Actualizar bordes del perímetro
    std::vector<unsigned int> edgeIndices;
    std::set<std::pair<unsigned int, unsigned int>> allPerimeterEdges;

    // Extraer bordes del perímetro de cada cara
    for (auto& face : model.getFaces()) {
        std::vector<std::pair<unsigned int, unsigned int>> facePerimeter = perimeterIndex(face);
        for (const auto& edge : facePerimeter) {
            // Ordenar para evitar duplicados
            unsigned int v1 = edge.first;
            unsigned int v2 = edge.second;
            if (v1 > v2) std::swap(v1, v2);
            allPerimeterEdges.insert({ v1, v2 });
        }
    }

    // Convertir a vector de índices
    for (const auto& edge : allPerimeterEdges) {
        edgeIndices.push_back(edge.first);
        edgeIndices.push_back(edge.second);
    }

    // Actualizar EBO de aristas
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.edgeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_DYNAMIC_DRAW);
    mesh.edgeIndexCount = (GLuint)edgeIndices.size();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Actualizar EBO (índices de triángulos)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.size() * sizeof(unsigned int), index.data(), GL_DYNAMIC_DRAW);
    mesh.indexCount = (GLuint)index.size();

   
}

// Calcula las normales de todos los vértices del modelo USAR SOLO PARA ILUMINACIÓN DE RENDERIZADO
void calculateNormals(Model& model) {
    std::vector<Vertex>& vertices = model.getVertex();
    std::vector<Face>& faces = model.getFaces();

    // Reset todas las normales a cero
    for (auto& vertex : vertices) {
        vertex.normal = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    // Calcular normales de cada cara y acumular en los vértices
    for (auto& face : faces) {
        for (auto& triangle : face.index) {
            if (triangle[0] >= vertices.size() ||
                triangle[1] >= vertices.size() ||
                triangle[2] >= vertices.size()) {
                continue; // Saltar triángulos inválidos
            }
			Vertex& vA = vertices[triangle[0]];
			Vertex& vB = vertices[triangle[1]];
			Vertex& vC = vertices[triangle[2]];

   //         if(model.getInsideVertex().count(triangle[0]) == 1){
			//	vA.normal = calculateFaceNormal(model, face); // Normal de la cara
   //             continue;
			//}
			glm::vec3 v0 = vertices[triangle[0]].pos;
            glm::vec3 v1 = vertices[triangle[1]].pos;
            glm::vec3 v2 = vertices[triangle[2]].pos;

            // Calcular vectores de arista
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;

            // Calcular normal de la cara (producto cruz)
            glm::vec3 faceNormal = glm::cross(edge1, edge2);

            // Solo añadir si la normal es válida (no cero)
            if (glm::length(faceNormal) > 0.0001f) {
                faceNormal = glm::normalize(faceNormal);

                // Acumular la normal en cada vértice del triángulo
                vertices[triangle[0]].normal += faceNormal;
                vertices[triangle[1]].normal += faceNormal;
                vertices[triangle[2]].normal += faceNormal;
            }
        }
    }

    // Normalizar todas las normales de vértices
    for (auto& vertex : vertices) {
        if (glm::length(vertex.normal) > 0.0001f) {
            vertex.normal = glm::normalize(vertex.normal);
        }
        else {
            // Si no se pudo calcular una normal válida, usar una por defecto
            vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }
    }
}
// Compila y adjunta un shader al programa
void addShader(GLuint shaderProgram, const char* shaderText, GLenum shaderType) {
    GLuint shaderObj = glCreateShader(shaderType);
    if (shaderObj == 0) {
        fprintf(stderr, "Error creación shader object de tipo %d \n", shaderType);
        exit(1);
    }
    const GLchar* p[1];
    *p = shaderText;
    GLint length[1];
    *length = strlen(shaderText);
    glShaderSource(shaderObj, 1, p, length);
    glCompileShader(shaderObj);
    GLint success;
    glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar InfoLog[1024];
        glGetShaderInfoLog(shaderObj, 1024, NULL, InfoLog);
        fprintf(stderr, "Error compiling shader type %d: '%s'\n", shaderType, InfoLog);
        exit(1);
    }
    glAttachShader(shaderProgram, shaderObj);
}

// Compila los shaders de vértices y fragmentos y obtiene la ubicación de la matriz de vista-proyección
void compileShaders(const char* vertexShaderFileName,const char* fragmentShaderFileName, GLint* viewProjectionLocationPtr) {
    GLuint shaderProgram = glCreateProgram();
    if (shaderProgram == 0) {
        fprintf(stderr, "Error creación shader program \n");
        exit(1);
    }
    std::string vertexShader, fragmentShader;
    if (!ReadFile(vertexShaderFileName, vertexShader)) {
        exit(1);
    }
    addShader(shaderProgram, vertexShader.c_str(), GL_VERTEX_SHADER);
    if (!ReadFile(fragmentShaderFileName, fragmentShader)) {
        exit(1);
    }
    addShader(shaderProgram, fragmentShader.c_str(), GL_FRAGMENT_SHADER);
    glLinkProgram(shaderProgram);
    GLint success = 0;
    GLchar ErrorLog[1024] = { 0 };
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (success == 0) {
        glGetProgramInfoLog(shaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
    }
    *viewProjectionLocationPtr = glGetUniformLocation(shaderProgram, "gViewProjection");
    if (*viewProjectionLocationPtr == -1) {
        fprintf(stderr, "Error getting uniform location of 'gViewProjection' \n");
        exit(1);
    }
    glValidateProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }
    glUseProgram(shaderProgram);
}

// Crea un modelo de flecha orientado en la posición indicada y con la dirección dada que devueve la punta de la flecha
glm::vec3 createArrow(Model& arrowModel, const glm::vec3& position,
    const glm::vec3& color, const glm::vec3& direction,
    const glm::quat& rotation) {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    float shaftLength = 0.5f;
    float shaftWidth = 0.03f;
    float shaftHeight = 0.03f;
    float pyramidBaseSize = 0.08f;
    float pyramidHeight = 0.2f;

    // Aplicar rotación a la dirección base
    glm::vec3 rotatedDirection = rotation * direction;

    glm::vec3 baseDir = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 targetDir = glm::normalize(rotatedDirection);

    glm::mat4 rotMat = glm::mat4(1.0f);

    if (glm::length(targetDir - baseDir) > 0.001f) {
        if (glm::length(targetDir + baseDir) < 0.001f) {
            rotMat = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        }
        else {
            glm::vec3 axis = glm::cross(baseDir, targetDir);
            float angle = acos(glm::dot(baseDir, targetDir));
            rotMat = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
        }
    }

    unsigned int prismBaseIndex = vertices.size();

    std::vector<glm::vec3> prismPoints = {

        glm::vec3(-shaftWidth / 2, -shaftHeight / 2, 0.0f),
        glm::vec3(shaftWidth / 2, -shaftHeight / 2, 0.0f),
        glm::vec3(shaftWidth / 2,  shaftHeight / 2, 0.0f),
        glm::vec3(-shaftWidth / 2,  shaftHeight / 2, 0.0f),

        glm::vec3(-shaftWidth / 2, -shaftHeight / 2, shaftLength),
        glm::vec3(shaftWidth / 2, -shaftHeight / 2, shaftLength),
        glm::vec3(shaftWidth / 2,  shaftHeight / 2, shaftLength),
        glm::vec3(-shaftWidth / 2,  shaftHeight / 2, shaftLength)
    };

    for (const auto& point : prismPoints) {
        glm::vec4 transformed = rotMat * glm::vec4(point, 1.0f);
        vertices.push_back(Vertex(
            transformed.x + position.x,
            transformed.y + position.y,
            transformed.z + position.z,
            color.r, color.g, color.b
        ));
    }

    faces.push_back(Face({ prismBaseIndex, prismBaseIndex + 1, prismBaseIndex + 2 }));
    faces.push_back(Face({ prismBaseIndex, prismBaseIndex + 2, prismBaseIndex + 3 }));

    faces.push_back(Face({ prismBaseIndex + 4, prismBaseIndex + 6, prismBaseIndex + 5 }));
    faces.push_back(Face({ prismBaseIndex + 4, prismBaseIndex + 7, prismBaseIndex + 6 }));

    faces.push_back(Face({ prismBaseIndex + 1, prismBaseIndex + 5, prismBaseIndex + 6 }));
    faces.push_back(Face({ prismBaseIndex + 1, prismBaseIndex + 6, prismBaseIndex + 2 }));

    faces.push_back(Face({ prismBaseIndex + 0, prismBaseIndex + 3, prismBaseIndex + 7 }));
    faces.push_back(Face({ prismBaseIndex + 0, prismBaseIndex + 7, prismBaseIndex + 4 }));

    faces.push_back(Face({ prismBaseIndex + 2, prismBaseIndex + 6, prismBaseIndex + 7 }));
    faces.push_back(Face({ prismBaseIndex + 2, prismBaseIndex + 7, prismBaseIndex + 3 }));

    faces.push_back(Face({ prismBaseIndex + 0, prismBaseIndex + 4, prismBaseIndex + 5 }));
    faces.push_back(Face({ prismBaseIndex + 0, prismBaseIndex + 5, prismBaseIndex + 1 }));

    unsigned int pyramidBaseIndex = vertices.size();

    std::vector<glm::vec3> pyramidBasePoints = {
        glm::vec3(-pyramidBaseSize / 2, -pyramidBaseSize / 2, shaftLength),
        glm::vec3(pyramidBaseSize / 2, -pyramidBaseSize / 2, shaftLength),
        glm::vec3(pyramidBaseSize / 2,  pyramidBaseSize / 2, shaftLength),
        glm::vec3(-pyramidBaseSize / 2,  pyramidBaseSize / 2, shaftLength)
    };

    for (const auto& point : pyramidBasePoints) {
        glm::vec4 transformed = rotMat * glm::vec4(point, 1.0f);
        vertices.push_back(Vertex(
            transformed.x + position.x,
            transformed.y + position.y,
            transformed.z + position.z,
            color.r, color.g, color.b
        ));
    }

    glm::vec4 pyramidTip = rotMat * glm::vec4(0.0f, 0.0f, shaftLength + pyramidHeight, 1.0f);
    unsigned int pyramidTipIndex = vertices.size();
    vertices.push_back(Vertex(
        pyramidTip.x + position.x,
        pyramidTip.y + position.y,
        pyramidTip.z + position.z,
        color.r, color.g, color.b
    ));

    faces.push_back(Face({ pyramidBaseIndex, pyramidBaseIndex + 1, pyramidTipIndex }));

    faces.push_back(Face({ pyramidBaseIndex + 1, pyramidBaseIndex + 2, pyramidTipIndex }));

    faces.push_back(Face({ pyramidBaseIndex + 2, pyramidBaseIndex + 3, pyramidTipIndex }));

    faces.push_back(Face({ pyramidBaseIndex + 3, pyramidBaseIndex, pyramidTipIndex }));


    faces.push_back(Face({ pyramidBaseIndex, pyramidBaseIndex + 2, pyramidBaseIndex + 1 }));
    faces.push_back(Face({ pyramidBaseIndex, pyramidBaseIndex + 3, pyramidBaseIndex + 2 }));

    arrowModel = Model(vertices, faces);
    //calculateNormals(arrowModel);

    return pyramidTip;
}
// Intersección rayo-vértice
// Devuelve true si el rayo pasa suficientemente cerca del vértice y está orientado hacia la normal
bool rayIntersectsVertex(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    Vertex vertex,
    float epsilon,
    float& outDistance)
{
    glm::vec3& vertexPos = vertex.pos;

	// Vector desde el vértice al origen del rayo
    glm::vec3 diff = vertexPos - rayOrigin;
	// Proyección del vector diff sobre la dirección del rayo
	glm::vec3 normalizedRayDir = glm::normalize(rayDir);
	float t = glm::dot(diff, normalizedRayDir); // distancia a lo largo del rayo

    if (t < 0.0f) return false; // detrás de cámara

    glm::vec3 closestPoint = rayOrigin + t * rayDir;
	// Distancia desde el punto más cercano del rayo al vértice
    float dist = glm::length(closestPoint - vertexPos);
    outDistance = t;
	// Vertice dentro del radio
    return dist <= epsilon;
}

glm::vec3 calculateFaceNormal(Model& model, const Face& face) {
    if (face.index.empty()) return glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 normal(0.0f);
    int validTriangles = 0;

    for (auto& triangle : face.index) {
        

        glm::vec3 v0 = model.getVertex()[triangle[0]].pos;
        glm::vec3 v1 = model.getVertex()[triangle[1]].pos;
        glm::vec3 v2 = model.getVertex()[triangle[2]].pos;

        // Calcular vectores de arista
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        // Calcular normal de la cara (producto cruz)
        glm::vec3 triangleNormal = glm::cross(edge1, edge2);

        // Solo sumar si la normal es válida
        if (glm::length(triangleNormal) > 0.0001f) {
            normal += triangleNormal; // NO NORMALIZAR
            validTriangles++;
        }
    }

    // Promedio de las normales de todos los triángulos válidos
    if (validTriangles > 0 && glm::length(normal) > 0.0001f) {
        return glm::normalize(normal);
    }

    // Normal por defecto si no se pudo calcular
    return glm::vec3(0.0f, 1.0f, 0.0f);
}

std::vector<std::pair<unsigned int, unsigned int>> perimeterIndex(Face& face) {
	std::set<std::pair<unsigned int, unsigned int>> edgeCount;
    for (const auto& triangle : face.index) {
        for (int i = 0; i < 3; i++) {
            unsigned int v1 = triangle[i];
            unsigned int v2 = triangle[(i + 1) % 3];
			// Para mantener el orden consistente en el set no usar swap
            auto edge1 = std::make_pair(v1, v2);
			auto edge2 = std::make_pair(v2, v1);
            if (edgeCount.find(edge1) != edgeCount.end()) {
                edgeCount.erase(edge1); // Borrar si ya existe (interior)
            }
            else if (edgeCount.find(edge2) != edgeCount.end()) {
				edgeCount.erase(edge2); // Borrar si ya existe (interior)
            }
            else {
                edgeCount.insert(edge1); // Insertar si no existe (posible perímetro)
            }
        }
    }

	std::vector< std::pair<unsigned int, unsigned int>> perimeterIndices(edgeCount.begin(), edgeCount.end());
	return perimeterIndices;
}

// Calcula el centroide de un modelo (media de sus vértices).
glm::vec3 centroidModel(Model& model) {
    glm::vec3 sum(0.0f);
    const auto& vertices = model.getVertex();
    if (vertices.empty()) return sum;
    for (const auto& v : vertices) sum += v.pos;
    return sum / (float)vertices.size();
}

glm::vec3 centroidFace(Model& model, Face& face) {
    glm::vec3 sum(0.0f);
    const auto& vertices = model.getVertex();
    if (face.index.empty()) return sum;
    // vertices unicos
    std::set<unsigned int> uniqueIndices;
    for (const auto& triangle : face.index) {
        for (const auto& index : triangle) {
            uniqueIndices.insert(index);
        }
    }
    for (const auto& index : uniqueIndices) {
        sum += vertices[index].pos;
    }
    return sum / (float)uniqueIndices.size();
}
