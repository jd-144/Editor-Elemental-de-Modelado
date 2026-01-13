#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <set>
#include <windows.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "util.h"
#include "camara.h"
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glut.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include <map>
std::vector<MeshBuffers> meshList;
std::vector<Model> modelList;

const char* VERTEXSHADERFILENAME = "shader.vert";
const char* FRAGMENTSHADERFILENAME = "shader.frag";

// Modos de interacción y selección
enum Mode { NONE_MODE, ROTATE, TRANSLATE, SCALE, SELECT, CREATE_VERTEX, EXTRUDE};
enum SelectionMode { UNKNOWN, MODEL, FACE, VERTEX };
enum GizmoAxis { NONE_AXIS, X_AXIS, Y_AXIS, Z_AXIS };

Mode currentMode = NONE_MODE;
SelectionMode currentSelection = UNKNOWN;

GizmoAxis selectedAxis = NONE_AXIS;
bool showGizmo = false;
Gizmo gizmo;
MeshBuffers gizmoMeshX, gizmoMeshY, gizmoMeshZ;

glm::vec3 gizmoPosition;

int width = 1920 / 2;
int height = 1080 / 2;
GLint viewProjectionLocation, lightPosLocation, viewPosLocation;
Camara* cam = nullptr;

// Variables de selección
size_t selectedModelPosition = -1;
size_t selectedFacePosition = -1;
unsigned int selectedTrianglePosition = -1;
size_t selectedVertexPosition = -1;
glm::vec3 lastIntersectionPoint;
glm::vec3 selectedVertexWorldPos;

bool createdVertexExtruded = false;
// Inicializa la cámara principal con parámetros por defecto.
static void initCamara() {
    if (cam) delete cam;
	float FOV = 45.0f;
	float zNear = 0.01f;
	float zFar = 100.0f;
	glm::vec3 pos(0.0f, 0.0f, 3.0f);
	glm::vec3 target(0.0f, 0.0f, 0.0f);
	glm::vec3 up(0.0f, 1.0f, 0.0f);
    float aspectRatio = (float)width / height;


    float tanHalfFov = std::tan(glm::radians(FOV) / 2.0f);


    float left = -tanHalfFov * zNear * aspectRatio;
    float right = tanHalfFov * zNear * aspectRatio;
    float bottom = -tanHalfFov * zNear;
    float top = tanHalfFov * zNear;

    cam = new Camara(pos, target, up, left, right, bottom, top, zNear, zFar);
}

// Proyecta las coordenadas del mouse a una esfera para rotación orbital.
static glm::vec3 mapToSphere(int x, int y) {
	float nx = (2.0f * x - width) / width; // Normaliza a [-1, 1]
	float ny = (height - 2.0f * y) / height; // Normaliza a [-1, 1], invierte Y
    float length = nx * nx + ny * ny;
    if (length <= 1.0f) {
		// Dentro de la esfera 
		float nz = sqrt(1.0f - length); // usa la fórmula de la esfera x^2 + y^2 + z^2 = 1
        return glm::vec3(nx, ny, nz);
    } else {
		// Fuera de la esfera, proyecta al borde de la esfera
        float norm = 1.0f / sqrt(length);
        return glm::vec3(nx * norm, ny * norm, 0.0f);
    }
}



// Crea los buffers de OpenGL para un modelo.
MeshBuffers createMesh(Model& model) {
    std::vector<Vertex> vertex = model.getVertex();
    std::vector<unsigned int> index = model.getAllIndex();
    MeshBuffers mesh;

    // Crear VAO principal para triángulos
    glGenVertexArrays(1, &mesh.VAO);
    glBindVertexArray(mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertex.size() * sizeof(Vertex), vertex.data(), GL_DYNAMIC_DRAW);
    glGenBuffers(1, &mesh.EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.size() * sizeof(unsigned int), index.data(), GL_DYNAMIC_DRAW);

    // Atributos: posición, normal, color
    glEnableVertexAttribArray(0); // inPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); // inNormal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); // inColor
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    mesh.indexCount = (GLuint)index.size();

    // Crear VAO para aristas
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

    glGenVertexArrays(1, &mesh.edgeVAO);
    glBindVertexArray(mesh.edgeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glGenBuffers(1, &mesh.edgeEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.edgeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // inPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); // inNormal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); // inColor
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));

    mesh.edgeIndexCount = (GLuint)edgeIndices.size();
    glBindVertexArray(0);

    return mesh;
}
// Libera los recursos de un mesh.
void deleteMesh(MeshBuffers& mesh) {
    if (mesh.VAO) { glDeleteVertexArrays(1, &mesh.VAO); mesh.VAO = 0; }
    if (mesh.VBO) { glDeleteBuffers(1, &mesh.VBO); mesh.VBO = 0; }
    if (mesh.EBO) { glDeleteBuffers(1, &mesh.EBO); mesh.EBO = 0; }
    if (mesh.edgeVAO) { glDeleteVertexArrays(1, &mesh.edgeVAO); mesh.edgeVAO = 0; }
    if (mesh.edgeEBO) { glDeleteBuffers(1, &mesh.edgeEBO); mesh.edgeEBO = 0; }
    mesh.indexCount = 0;
    mesh.edgeIndexCount = 0;
}

// Elimina todos los meshes de la lista global.
static void cleanupAllMeshes() {
    for (auto& mesh : meshList) deleteMesh(mesh);
    meshList.clear();
}



// Crea un nuevo vértice en la cara seleccionada del modelo seleccionado.
static void createVertex() {
    if (selectedModelPosition !=-1 && selectedFacePosition != -1) {
		// Se añade un nuevo vértice en el punto de intersección uniendolo a los vértices del triángulo seleccionado
        Model& model = modelList[selectedModelPosition];
        Face& face = model.getFaces()[selectedFacePosition];
        subdivideFace(model, face, lastIntersectionPoint, selectedTrianglePosition);
        updateModelMesh(model, meshList[selectedModelPosition]);

    }
}



// Detecta si el usuario ha hecho clic sobre algún eje del gizmo de transformación.
static bool detectGizmoClick(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) {
    if (!showGizmo || !(currentMode == TRANSLATE || currentMode == ROTATE || currentMode == SCALE || currentMode == EXTRUDE)) return false;
    float closestT = FLT_MAX;
    GizmoAxis hitAxis = NONE_AXIS;
    // Función lambda para comprobar intersección con cada modelo del gizmo
    auto checkGizmoModel = [&](Model& gizmoModel, GizmoAxis axis) {
        for (int fi = 0; fi < gizmoModel.getFaces().size(); ++fi) {
            Face& face = gizmoModel.getFaces()[fi];
            unsigned int trianglePosition = -1;
            float t = -1.0f;
            // Comprueba intersección rayo-cara
            glm::vec3* intersectionPoint = rayIntersectsFace(rayOrigin, rayDirection, gizmoModel, face, trianglePosition, t);
            if (intersectionPoint) {
				hitAxis = axis; // Guarda el eje detectado
				break; // Sale del bucle al detectar el primer eje intersectado
            }
            delete intersectionPoint;
        }
    };
    // Comprueba los tres ejes del gizmo
    checkGizmoModel(gizmo.modelX, X_AXIS);
    checkGizmoModel(gizmo.modelY, Y_AXIS);
    checkGizmoModel(gizmo.modelZ, Z_AXIS);
    // Si se detectó un eje, lo selecciona
    if (hitAxis != NONE_AXIS) {
        selectedAxis = hitAxis;
        return true;
    }
    return false;
}

// Busca el vértice más cercano al rayo lanzado desde la cámara.
static bool findClosestVertex(const glm::vec3& rayOrigin, const glm::vec3& rayDirection,
    Model*& outModel, Vertex*& outVertex, size_t& outModelIndex,
    size_t& outVertexIndex, glm::vec3& outWorldPos) {
    float closestDistance = FLT_MAX;
    Model* closestModel = nullptr;
    Vertex* closestVertex = nullptr;
    size_t closestModelIndex = -1;
    size_t closestVertexIndex = -1;
    glm::vec3 closestWorldPos;
    // Recorre todos los modelos
    for (size_t mi = 0; mi < modelList.size(); ++mi) {
        Model& model = modelList[mi];
        std::vector<Vertex>& vertices = model.getVertex();
        // Recorre todos los vértices del modelo
        for (size_t vi = 0; vi < vertices.size(); ++vi) {
            Vertex& vertex = vertices[vi];
            float distance = -1.0f;
            // Comprueba si el rayo pasa cerca del vértice
            if (rayIntersectsVertex(rayOrigin, rayDirection, vertex, 0.1f, distance)) {
                // Si es el más cercano hasta ahora, lo guarda
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestModel = &model;
                    closestVertex = &vertex;
                    closestModelIndex = mi;
                    closestVertexIndex = vi;
                    closestWorldPos = vertex.pos;
                }
            }
        }
    }
    // Si se encontró un vértice cercano, actualiza los parámetros de salida
    if (closestModel) {
        outModel = closestModel;
        outVertex = closestVertex;
        outModelIndex = closestModelIndex;
        outVertexIndex = closestVertexIndex;
        outWorldPos = closestWorldPos;
        return true;
    }
    return false;
}

// Busca la intersección más cercana entre el rayo y las caras de todos los modelos.
static bool findClosestFaceIntersection(const glm::vec3& rayOrigin, const glm::vec3& rayDirection,
    Model*& outModel, Face*& outFace, size_t& outModelIndex,
    size_t& outFaceIndex, unsigned int& outTriangleIndex,
    glm::vec3& outIntersectionPoint) {
    float closestT = FLT_MAX;
    Model* closestModel = nullptr;
    Face* closestFace = nullptr;
    size_t closestModelIndex = -1;
    size_t closestFaceIndex = -1;
    unsigned int closestTriangleIndex = -1;
    glm::vec3 closestIntersection;
    // Recorre todos los modelos
    for (size_t mi = 0; mi < modelList.size(); ++mi) {
        Model& model = modelList[mi];
        std::vector<Face>& faces = model.getFaces();
        // Recorre todas las caras del modelo
        for (size_t fi = 0; fi < faces.size(); ++fi) {
            Face& face = faces[fi];
            unsigned int trianglePosition = -1;
            float t = -1.0f;
            // Comprueba intersección rayo-cara
            glm::vec3* intersectionPoint = rayIntersectsFace(rayOrigin, rayDirection, model, face, trianglePosition, t);
            if (intersectionPoint && t < closestT) {
                closestT = t;
                closestIntersection = *intersectionPoint;
                closestModel = &model;
                closestFace = &face;
                closestModelIndex = mi;
                closestFaceIndex = fi;
                closestTriangleIndex = trianglePosition;
            }
            delete intersectionPoint;
        }
    }
    // Si se encontró una intersección, actualiza los parámetros de salida
    if (closestModel) {
        outModel = closestModel;
        outFace = closestFace;
        outModelIndex = closestModelIndex;
        outFaceIndex = closestFaceIndex;
        outTriangleIndex = closestTriangleIndex;
        outIntersectionPoint = closestIntersection;
        return true;
    }
    return false;
}

// Actualiza la posición y modelos del gizmo según la selección.
static void updateGizmoForSelectedModel() {
    if ((selectedModelPosition != -1 && currentSelection == MODEL &&
        (currentMode == TRANSLATE || currentMode == SCALE || currentMode == EXTRUDE)) ||
        (selectedVertexPosition != -1 && currentSelection == VERTEX && currentMode == TRANSLATE) ||
        (selectedFacePosition!= -1 && currentSelection == FACE && currentMode == EXTRUDE)) {

        showGizmo = true;

        if (currentSelection == VERTEX && selectedVertexPosition != -1) {
            gizmoPosition = selectedVertexWorldPos;
        }
        else if (currentSelection == FACE && selectedFacePosition != -1 && selectedModelPosition != -1) {
            // Para extrusión, posicionar el gizmo en el centro de la cara y orientarlo según la normal
			Model& selectedModel = modelList[selectedModelPosition];
			Face& selectedFace = selectedModel.getFaces()[selectedFacePosition];

			gizmoPosition = centroidFace(selectedModel, selectedFace);
			glm::vec3 normal = calculateFaceNormal(selectedModel, selectedFace);
            gizmo.setRotation(quatFromVectors(glm::vec3(0.0f, 1.0f, 0.0f), normal));
        }
        else if (selectedModelPosition != -1 && currentSelection == MODEL) {
            gizmoPosition = centroidModel(modelList[selectedModelPosition]);
        }

        gizmo = Gizmo(gizmoPosition);
        deleteMesh(gizmoMeshX);
        deleteMesh(gizmoMeshY);
        deleteMesh(gizmoMeshZ);
        gizmoMeshX = createMesh(gizmo.modelX);
        gizmoMeshY = createMesh(gizmo.modelY);
        gizmoMeshZ = createMesh(gizmo.modelZ);
    }
    else {
        showGizmo = false;
    }
}
// Reinicia todas las variables de selección.
static void resetSelection() {
    selectedModelPosition = -1;
    selectedFacePosition = -1;
    selectedVertexPosition = -1;
    selectedTrianglePosition = -1;
    selectedAxis = NONE_AXIS;
}
static void gizmoLookAtNormal() {
    if (showGizmo && selectedFacePosition != -1 && selectedModelPosition != -1) {
        Model& selectedModel = modelList[selectedModelPosition];
        Face& selectedFace = selectedModel.getFaces()[selectedFacePosition];

		glm::vec3 normal = calculateFaceNormal(selectedModel, selectedFace);
		glm::vec3 newPosition = centroidFace(selectedModel, selectedFace);

        // Only update if position or normal has actually changed
        if (glm::length(newPosition - gizmoPosition) > 0.001f ||
            glm::length(normal - gizmo.getCurrentNormal()) > 0.001f) {

            gizmo.setRotation(quatFromVectors(glm::vec3(0.0f, 1.0f, 0.0f), normal));
            gizmo.setNormal(normal); // Store the current normal
            gizmoPosition = newPosition;
            gizmo.setPosition(gizmoPosition);

            deleteMesh(gizmoMeshX);
            deleteMesh(gizmoMeshY);
            deleteMesh(gizmoMeshZ);
            gizmoMeshX = createMesh(gizmo.modelX);
            gizmoMeshY = createMesh(gizmo.modelY);
            gizmoMeshZ = createMesh(gizmo.modelZ);
        }
    }
}

// Realiza la selección de modelo, cara o vértice según la posición del mouse en pantalla.
static void select(int x, int y) {

    if (currentSelection != FACE &&
        currentSelection != MODEL && currentSelection != VERTEX) {
        return;
    }

	// Calcula el rayo desde la cámara a la pantalla usando la matriz de vista y proyección
    glm::vec3 rayOrigin = cam->getPosition();
    glm::vec3 rayDirection = screenToWorld(x, y, cam->getViewAndProjectionMatrix(), width, height);

    // Si se hace clic en el gizmo, selecciona el eje y termina
    if (detectGizmoClick(rayOrigin, rayDirection)) return;

    // Busca el vértice más cercano al rayo
    Model* closestModel = nullptr;
    Vertex* closestVertex = nullptr;
    size_t closestModelIndex = -1;
    size_t closestVertexIndex = -1;
    glm::vec3 closestWorldPos;

    bool foundVertex = findClosestVertex(rayOrigin, rayDirection, closestModel, closestVertex,
                                         closestModelIndex, closestVertexIndex, closestWorldPos);

    // Si se encuentra un vértice cercano, actualiza la selección de vértice
    if (foundVertex && selectedModelPosition != -1) {
        bool isNewModel = (closestModelIndex != selectedModelPosition);

        if (isNewModel) {
            resetSelection();
            selectedModelPosition = closestModelIndex;
        }
        selectedVertexPosition = closestVertexIndex;
        selectedVertexWorldPos = closestWorldPos;
        selectedAxis = NONE_AXIS;
        currentSelection = VERTEX;

        updateGizmoForSelectedModel();

        return;
    }

    // Si no hay vértice, busca la cara más cercana
    Face* closestFace = nullptr;
    size_t closestFaceIndex = -1;
    unsigned int closestTriangleIndex = -1;
    glm::vec3 closestIntersection;

    bool foundFace = findClosestFaceIntersection(rayOrigin, rayDirection, closestModel, closestFace,
                                                 closestModelIndex, closestFaceIndex, closestTriangleIndex,
                                                 closestIntersection);

    if (foundFace) {
        bool isNewModel = (closestModelIndex != selectedModelPosition);
        bool isNewFace = (closestFaceIndex != selectedFacePosition);

        selectedModelPosition = closestModelIndex;
        selectedFacePosition = closestFaceIndex;
        selectedTrianglePosition = closestTriangleIndex;
        lastIntersectionPoint = closestIntersection;
        selectedAxis = NONE_AXIS;
        currentSelection = MODEL;

        // Si se selecciona un modelo diferente o había un vértice seleccionado, actualiza el gizmo
        if (isNewModel || selectedVertexPosition != -1) {
            selectedVertexPosition = -1;
            updateGizmoForSelectedModel();
        }
        if (isNewFace && currentMode == EXTRUDE) {
            gizmoLookAtNormal();
			createdVertexExtruded = false;
		}
    } else {
        // Si no se selecciona nada, reinicia la selección
        if(currentMode!= SCALE){
            resetSelection();
            showGizmo = false;
            
        }
        selectedAxis = NONE_AXIS;
        
    }
}

static glm::vec3 movementGizmo(float deltaX, float deltaY) {
    glm::mat3 gizmoBasisMatrix = glm::mat3_cast(gizmo.rotation);
    glm::vec3 gizmoX = gizmoBasisMatrix * glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 gizmoY = gizmoBasisMatrix * glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 gizmoZ = gizmoBasisMatrix * glm::vec3(0.0f, 0.0f, 1.0f);

    glm::vec3 cameraForward = cam->getFront();
    glm::vec3 cameraRight = cam->getRight();
    glm::vec3 cameraUp = cam->getUp();

    
    
    glm::vec3 gizmoWorldPos = gizmo.position;
    glm::vec3 selectedAxisDir;
    switch (selectedAxis) {
    case X_AXIS: 
        selectedAxisDir = gizmoX;

        break;
    case Y_AXIS: 
        selectedAxisDir = gizmoY;
        break;
    case Z_AXIS: 
        selectedAxisDir = gizmoZ;
        break;
    default: return glm::vec3(0.0f);
    }
	// Verificar si la cámara está demasiado cerca del gizmo
    bool tooNear = glm::length(gizmoWorldPos + selectedAxisDir - cam->getPosition()) < 0.5f;

	

    glm::vec3 camToGizmo = glm::normalize(gizmoWorldPos - cam->getPosition());
    float facingDot = glm::abs(glm::dot(camToGizmo, cameraForward));
    bool isCentered = facingDot > 0.999f;

    // Verificar si el eje está casi paralelo a la cámara
    float thresholdParallelMovement = 0.85f;
    float dotProduct = glm::abs(glm::dot(glm::normalize(selectedAxisDir), glm::normalize(cameraForward)));
    bool isAxisParallelToCamera = dotProduct > thresholdParallelMovement;

    if (isCentered && isAxisParallelToCamera) {
        // Se mueve en el eje pero no se puede calcular la dirección (se proyectaría en 2D como un punto)
        // El movimiento pasa a ser el delta en el eje Y de la pantalla (2D)

		// Si la cámara está demasiado cerca y el movimiento es hacia la cámara, no se permite mover
        if (tooNear && deltaY >= 0.0f) return glm::vec3(0.0f);

        return selectedAxisDir * deltaY * 0.01f;
    }
    

    // Proyectamos el centro del gizmo y la punta del eje en 2D
    glm::vec4 clipCenter = cam->getViewAndProjectionMatrix() * glm::vec4(gizmoWorldPos, 1.0);
    glm::vec4 clipTip = cam->getViewAndProjectionMatrix() * glm::vec4(gizmoWorldPos + selectedAxisDir, 1.0);

    clipCenter /= clipCenter.w;
    clipTip /= clipTip.w;

    // Direccion del eje proyectado en la pantalla
    glm::vec2 axisScreenDir = glm::normalize(glm::vec2(clipTip) - glm::vec2(clipCenter));

    
    float ndcDeltaX = (2.0f * deltaX) / (width);
    float ndcDeltaY = (-2.0f * deltaY) / (height);

    glm::vec2 mouseMoveDir = glm::normalize(glm::vec2(ndcDeltaX, ndcDeltaY));
    float alignment = glm::dot(axisScreenDir, mouseMoveDir);

	// Si la cámara está demasiado cerca y el movimiento es hacia la cámara, no se permite mover
    if (tooNear && (std::isnan(alignment) || alignment >= 0.0f)) return glm::vec3(0.0f);



    // Magnitud del movimiento
    float movementLength = glm::length(glm::vec2(ndcDeltaX, ndcDeltaY));

    float signedMovement = movementLength * (alignment >= 0.0f ? 1.0f : -1.0f);

    return selectedAxisDir * signedMovement;
}

static void translate(float deltaX, float deltaY) {
    if ((selectedModelPosition == -1 && selectedVertexPosition == -1) || selectedAxis == NONE_AXIS) return;

	glm::vec3 translation = movementGizmo(deltaX, deltaY);
    
    // Aplica la traslación al vértice seleccionado
    if (currentSelection == VERTEX && selectedVertexPosition != -1 && selectedModelPosition != -1) {
		Model* selectedModel = &modelList[selectedModelPosition];
		Vertex* selectedVertex = &selectedModel->getVertex()[selectedVertexPosition];
        selectedVertex->pos += translation;
        selectedVertexWorldPos = selectedVertex->pos;
        updateModelMesh(*selectedModel, meshList[selectedModelPosition]);
    }
    // O aplica la traslación a todos los vértices del modelo seleccionado
    else if (currentSelection == MODEL && selectedModelPosition != -1) {
        Model* selectedModel = &modelList[selectedModelPosition];
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), translation);
        for (auto& v : selectedModel->getVertex()) {
            v.pos = glm::vec3(translationMatrix * glm::vec4(v.pos, 1.0f));
        }
        updateModelMesh(*selectedModel, meshList[selectedModelPosition]);
    }
    // Actualiza la posición del gizmo y sus modelos
    gizmoPosition += translation;
    gizmo.setPosition(gizmoPosition);
    deleteMesh(gizmoMeshX);
    deleteMesh(gizmoMeshY);
    deleteMesh(gizmoMeshZ);
    gizmoMeshX = createMesh(gizmo.modelX);
    gizmoMeshY = createMesh(gizmo.modelY);
    gizmoMeshZ = createMesh(gizmo.modelZ);
}

static void scale(float deltaX, float deltaY) {
    if (selectedModelPosition == -1) return;
    
    // Obtiene la dirección del eje activo (transformada por el gizmo)
    glm::mat3 gizmoBasisMatrix = glm::mat3_cast(gizmo.rotation);
    glm::vec3 axisDir;
    switch (selectedAxis) {
    case X_AXIS: axisDir = gizmoBasisMatrix * glm::vec3(1, 0, 0); break;
    case Y_AXIS: axisDir = gizmoBasisMatrix * glm::vec3(0, 1, 0); break;
    case Z_AXIS: axisDir = gizmoBasisMatrix * glm::vec3(0, 0, 1); break;
    }
    // Centro de escala
	Model* selectedModel = &modelList[selectedModelPosition];
    glm::vec3 center = centroidModel(*selectedModel);
    float delta;
    glm::vec3 scaleVec(1.0f);

    if (selectedAxis == NONE_AXIS) {
		float scaleFactor = 1.0f - deltaY * 0.01f; // Escala uniforme según movimiento Y del ratón

        // Evita escala negativa o demasiado pequeña
        if (scaleFactor < 0.01f) scaleFactor = 0.01f;
        scaleVec = glm::vec3(scaleFactor);
    }
    else {
        std::cout << "Scaling along axis\n";
        // Calcula desplazamiento proyectado sobre el eje seleccionado
        glm::vec3 move = movementGizmo(deltaX, deltaY);
        float delta = glm::dot(move, axisDir); // componente sobre el eje
        float scaleFactor = 1.0f + delta * 0.5f;

        // Evita escala negativa o demasiado pequeña
        if (scaleFactor < 0.01f) scaleFactor = 0.01f;

        // factor de escala por eje
        
        if (selectedAxis == X_AXIS) scaleVec.x = scaleFactor;
        else if (selectedAxis == Y_AXIS) scaleVec.y = scaleFactor;
        else if (selectedAxis == Z_AXIS) scaleVec.z = scaleFactor;
    }
    
    glm::mat4 gizmoRot = glm::mat4_cast(gizmo.rotation);
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), -center);

    glm::mat4 finalTransform =
        glm::translate(glm::mat4(1.0f), center) *
        gizmoRot *
        glm::scale(glm::mat4(1.0f), scaleVec) *
        glm::transpose(gizmoRot) *
        transform;

    // aplica a vértices
    for (auto& v : selectedModel->getVertex())
        v.pos = glm::vec3(finalTransform * glm::vec4(v.pos, 1.0f));

    // Actualiza buffers
    updateModelMesh(*selectedModel, meshList[selectedModelPosition]);

    // Reubica gizmo
    gizmoPosition = centroidModel(*selectedModel);
    gizmo.setPosition(gizmoPosition);

    deleteMesh(gizmoMeshX);
    deleteMesh(gizmoMeshY);
    deleteMesh(gizmoMeshZ);
    gizmoMeshX = createMesh(gizmo.modelX);
    gizmoMeshY = createMesh(gizmo.modelY);
    gizmoMeshZ = createMesh(gizmo.modelZ);
}

static void extrudeFace(float deltaX, float deltaY) {
    if (selectedModelPosition == -1 || selectedFacePosition == -1) return;

	Model& model = modelList[selectedModelPosition];
	Face* selectedFace = &model.getFaces()[selectedFacePosition];
    glm::vec3 move = movementGizmo(deltaX, deltaY);
    std::vector<Vertex>& vertices = model.getVertex();

    if (!createdVertexExtruded) {
        std::map<unsigned int, unsigned int> oldToNewVertexMap;
        std::set<unsigned int> uniqueOldVertexIndices;

        // Obtener vértices únicos de la cara original ANTES de modificar el vector
        for (int i = 0; i < selectedFace->index.size(); i++) {
            for (int j = 0; j < 3; j++) {
                uniqueOldVertexIndices.insert(selectedFace->index[i][j]);
            }
        }

        // Crear nuevos vértices
        std::vector<unsigned int> newVertexIndices;
        for (auto idx : uniqueOldVertexIndices) {
            Vertex newV = vertices[idx];
            newV.pos += move;
            vertices.push_back(newV);
            newVertexIndices.push_back(vertices.size() - 1);
            oldToNewVertexMap[idx] = vertices.size() - 1;
        }

        // Crear caras laterales
        std::vector<std::pair<unsigned int, unsigned int>> perimeterEdges = perimeterIndex(*selectedFace);
        std::vector<Face> newSideFaces; // Almacenar temporalmente

        for (const auto& edge : perimeterEdges) {
            unsigned int v0_old = edge.first;
            unsigned int v1_old = edge.second;
            unsigned int v0_new = oldToNewVertexMap[v0_old];
            unsigned int v1_new = oldToNewVertexMap[v1_old];

            Face sideFace;
            sideFace.index.push_back({ v0_old, v1_old, v1_new });
            sideFace.index.push_back({ v0_old, v1_new, v0_new });
            newSideFaces.push_back(sideFace);
        }

        // Agregar todas las caras laterales de una vez
        for (auto& sideFace : newSideFaces) {
            model.getFaces().push_back(sideFace);
        }

        // Reestablecer el puntero después de modificar el vector
        selectedFace = &model.getFaces()[selectedFacePosition];
        // Modificar la cara original para usar los nuevos vértices
        for (int i = 0; i < selectedFace->index.size(); i++) {
            for (int j = 0; j < 3; j++) {
                unsigned int oldIndex = selectedFace->index[i][j];
                selectedFace->index[i][j] = oldToNewVertexMap[oldIndex];
            }

        }

        createdVertexExtruded = true;

    }
    else {
        // Mover vértices existentes
        std::set<unsigned int> uniqueNewVertexIndices;
        for (int i = 0; i < selectedFace->index.size(); i++) {
            for (int j = 0; j < 3; j++) {
                uniqueNewVertexIndices.insert(selectedFace->index[i][j]);
            }
        }
        for (const auto& vertexIndex : uniqueNewVertexIndices) {
            vertices[vertexIndex].pos += move;
        }
        gizmoPosition += move;
    }

    // actualizar buffers y gizmo
    updateModelMesh(model, meshList[selectedModelPosition]);
    gizmo.setPosition(gizmoPosition);

    deleteMesh(gizmoMeshX);
    deleteMesh(gizmoMeshY);
    deleteMesh(gizmoMeshZ);
    gizmoMeshX = createMesh(gizmo.modelX);
    gizmoMeshY = createMesh(gizmo.modelY);
    gizmoMeshZ = createMesh(gizmo.modelZ);
}

static size_t createPyramid(int sides) {
    if (!cam || sides < 3) return -1;
    
    glm::vec3 cameraPos = cam->getPosition();
    glm::vec3 cameraFront = cam->getFront();
    glm::vec3 pyramidPosition = cameraPos + cameraFront * 3.0f;
    
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    
    float radius = 1.0f;
    float height = 2.0f;
    
    // Vértices del perímetro de la base
    for (int i = 0; i < sides; ++i) {
        float angle = 2.0f * glm::pi<float>() * i / sides;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        vertices.push_back(Vertex(x, -height/2.0f, z));
    }
    
    // Vértice de la cúspide
    vertices.push_back(Vertex(0.0f, height/2.0f, 0.0f));
    unsigned int apexIndex = sides;
    
    // Caras laterales
    for (unsigned int i = 0; i < (unsigned int)sides; ++i) {
        unsigned int next_i = (i + 1) % sides;
        
        // Vértices de la base
        unsigned int base_current = i;
        unsigned int base_next = next_i;
        
        // Cara lateral
        faces.push_back({
            apexIndex,
            base_next,
            base_current
        });
    }
    
	// Base Conjunto de triangulos en un sola cara
	std::vector<std::array<unsigned int, 3>>baseTriangles;
    for (unsigned int i = 1; i < (unsigned int)sides - 1; ++i) {
        baseTriangles.push_back({
            0,
            i,
            i +1
			});
    }
    faces.push_back(baseTriangles);
    
    // Aplicar transformación
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), pyramidPosition)*glm::mat4_cast(cam->getOrientation());
    transform = glm::scale(transform, glm::vec3(0.5f, 0.5f, 0.5f));
    
    for (auto& vertex : vertices) {
        glm::vec4 transformedPos = transform * glm::vec4(vertex.pos, 1.0f);
        vertex.pos = glm::vec3(transformedPos);
    }
    
    Model pyramidModel(vertices, faces);
    calculateNormals(pyramidModel);
    modelList.push_back(pyramidModel);
    meshList.push_back(createMesh(pyramidModel));
    
    return modelList.size() - 1;
}
static size_t createPrism(unsigned int sides) {
    if (!cam || sides < 3) return -1;
    
    // Obtener posición y dirección frontal de la cámara
    glm::vec3 cameraPos = cam->getPosition();
    glm::vec3 cameraFront = cam->getFront();
    
    // Calcular posición del prisma: posición de cámara + dirección frontal
    glm::vec3 prismPosition = cameraPos + cameraFront * 3.0f;
    
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    
    // Crear vértices de la base inferior
    float radius = 1.0f;
    float height = 2.0f;
    
    // Base inferior
    for (int i = 0; i < sides; ++i) {
        float angle = 2.0f * glm::pi<float>() * i / sides;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        vertices.push_back(Vertex(x, -height/2.0f, z));
    }
    
    // Base superior
    for (int i = 0; i < sides; ++i) {
        float angle = 2.0f * glm::pi<float>() * i / sides;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        vertices.push_back(Vertex(x, height/2.0f, z));
    }
    
    // Caras laterales
    for (unsigned int i = 0; i < (unsigned int)sides; ++i) {
        unsigned int next_i = (i + 1) % sides;
        unsigned int bottom_current = i;
        unsigned int bottom_next = next_i;
        unsigned int top_current = i + sides;
        unsigned int top_next = next_i + sides;
        
        std::vector<std::array<unsigned int, 3>>lateralTriangles;

        // Caras laterales
        lateralTriangles.push_back({
            bottom_current, 
            top_next, 
            bottom_next
        });
        lateralTriangles.push_back({
            bottom_current, 
            top_current, 
            top_next
        });
		faces.push_back(lateralTriangles);
    }
    
    // Base inferior
    std::vector<std::array<unsigned int, 3>>baseTriangles;
    for (unsigned int i = 1; i < (unsigned int)sides - 1; ++i) {
        baseTriangles.push_back({
            0,
            i,
            i + 1
            });
    }
    faces.push_back(baseTriangles);
    
    // Base superior
    // Base Conjunto de triangulos en un sola cara
    std::vector<std::array<unsigned int, 3>>baseUpTriangles;
    for (unsigned int i = 1; i < (unsigned int)sides - 1; ++i) {
        baseUpTriangles.push_back({
            sides,
            sides+i +1,
            sides+i
            });
    }
    faces.push_back(baseUpTriangles);
    
    // Aplicar transformación a la posición deseada
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), prismPosition) * glm::mat4_cast(cam->getOrientation());
    transform = glm::scale(transform, glm::vec3(0.5f, 0.5f, 0.5f));
    
    for (auto& vertex : vertices) {
        glm::vec4 transformedPos = transform * glm::vec4(vertex.pos, 1.0f);
        vertex.pos = glm::vec3(transformedPos);
    }
    
    Model prismModel(vertices, faces);
    calculateNormals(prismModel);
    modelList.push_back(prismModel);
    meshList.push_back(createMesh(prismModel));
    
    return modelList.size() - 1;
}
static size_t createCubes() {
    if (!cam) return -1;

    // Obtener posición y dirección frontal de la cámara
    glm::vec3 cameraPos = cam->getPosition();
    glm::vec3 cameraFront = cam->getFront();

    // Calcular posición del cubo: posición de cámara + dirección frontal
    glm::vec3 cubePosition = cameraPos + cameraFront * 3.0f; // 3.0f unidades adelante

    std::vector<Vertex> cubeVertices = {
        Vertex(-1.0f, -1.0f,  1.0f), Vertex(1.0f, -1.0f,  1.0f),
        Vertex(1.0f,  1.0f,  1.0f), Vertex(-1.0f,  1.0f,  1.0f),
        Vertex(-1.0f, -1.0f, -1.0f), Vertex(1.0f, -1.0f, -1.0f),
        Vertex(1.0f,  1.0f, -1.0f), Vertex(-1.0f,  1.0f, -1.0f)
    };

    std::vector<Face> faces = {
        {0, 1, 2, 2, 3, 0}, {1, 5, 6, 6, 2, 1}, {5, 4, 7, 7, 6, 5},
        {4, 0, 3, 3, 7, 4}, {3, 2, 6, 6, 7, 3}, {4, 5, 1, 1, 0, 4}
    };

    // Crear solo un cubo en la posición calculada
    std::vector<Vertex> transformedVertices;
    float scale = 0.5f; // Tamaño del cubo

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), cubePosition) * glm::mat4_cast(cam->getOrientation());
    transform = glm::scale(transform, glm::vec3(scale, scale, scale));

    for (const auto& v : cubeVertices) {
        glm::vec4 transformedPos = transform * glm::vec4(v.pos, 1.0f);
        Vertex transformedVertex(
            transformedPos.x,
            transformedPos.y,
            transformedPos.z
        );
        transformedVertices.push_back(transformedVertex);
    }

    Model cubeModel(transformedVertices, faces);
    calculateNormals(cubeModel);
    modelList.push_back(cubeModel);
    meshList.push_back(createMesh(cubeModel));

    // Devolver la posición del nuevo modelo en la lista
    return modelList.size() - 1;
}


// Callback de teclado para cambiar modos y mover la cámara.
static void keyboardCB(unsigned char key, int x, int y) {
    float cameraSpeed = 0.1f;
    if (!cam) return;
    glm::vec3 moveDir(0.0f);
    switch (key) {
    case 'w': moveDir = cam->getFront() * cameraSpeed; break;
    case 's': moveDir = -cam->getFront() * cameraSpeed; break;
    case 'a': moveDir = -cam->getRight() * cameraSpeed; break;
    case 'd': moveDir = cam->getRight() * cameraSpeed; break;
    case 'q': moveDir = -cam->getUp() * cameraSpeed; break;
    case 'e': moveDir = cam->getUp() * cameraSpeed; break;
    case 27: exit(0); break; // ESC
    case 'c': initCamara(); resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; break;
    case 'f': currentMode = SELECT; currentSelection = FACE; resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; break;
    case 'm': currentMode = SELECT; currentSelection = MODEL; resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; break;
    case 'p': currentMode = CREATE_VERTEX; currentSelection = FACE; resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; break;
    case 'r': if (currentMode == ROTATE) { currentMode = SELECT; }
            else { currentMode = ROTATE; currentSelection = MODEL; resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; } break;
    case 't': if (currentMode == TRANSLATE) { currentMode = SELECT; showGizmo = false; selectedAxis = NONE_AXIS; }
            else { currentMode = TRANSLATE; currentSelection = MODEL; } break;
    case 'v': currentMode = SELECT; currentSelection = VERTEX; resetSelection(); showGizmo = false; selectedAxis = NONE_AXIS; break;

    case 'b': {
        size_t newModelIndex = createCubes();
        if (newModelIndex != -1) {
            // Seleccionar el nuevo modelo
            selectedModelPosition = newModelIndex;

            // Cambiar a modo traslación
            currentMode = TRANSLATE;
            currentSelection = MODEL;

            // Actualizar gizmo para el nuevo modelo
            updateGizmoForSelectedModel();
        };
        break;
    }

    case 'n': gizmoLookAtNormal();break;

    case 'x':
        if (currentMode == EXTRUDE) {
            currentMode = SELECT;
            showGizmo = false;
            selectedAxis = NONE_AXIS;
        }
        else {
            currentMode = EXTRUDE;
            currentSelection = FACE;
            updateGizmoForSelectedModel();
        }
        break;
    }
    cam->move(moveDir);
}
// Callback de ratón para gestionar clicks y selección.
static void mouseCB(int button, int state, int x, int y) {
    ImGui_ImplGLUT_MouseFunc(button, state, x, y);
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    if (button == GLUT_LEFT_BUTTON && cam) {
        if (state == GLUT_DOWN) {
            cam->setLeftButtonPressed(true);
            cam->setRightButtonPressed(false);
            cam->setLastMousePos(mapToSphere(x, y));
            cam->setLastMouseX(x);
            cam->setLastMouseY(y);
            select(x, y);
            if (currentMode == CREATE_VERTEX && selectedModelPosition != -1 && selectedFacePosition != -1) {
                createVertex();
            }
        } else if (state == GLUT_UP) {
            cam->setLeftButtonPressed(false);
        }
    }
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) {
            cam->setRightButtonPressed(true);
            cam->setLeftButtonPressed(false);
            cam->setLastMousePos(mapToSphere(x, y));
        } else if (state == GLUT_UP) {
            cam->setRightButtonPressed(false);
        }
    }
}

// Callback de movimiento de ratón para rotar cámara, modelos y gizmo, y traslación con el gizmo.
static void mouseMotionCB(int x, int y) {
    ImGui_ImplGLUT_MotionFunc(x, y);
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    if (!cam) return;

    glm::vec3 currentPos = mapToSphere(x, y);
    glm::vec3 lastPos = cam->getLastMousePos();
    if (currentPos == lastPos) return;


    if (cam->isLeftButtonPressed()) {
        // Si está en modo traslación y hay un eje seleccionado, mueve el modelo/vértice
        if (currentMode == TRANSLATE && selectedAxis != NONE_AXIS) {

            // Calcula el movimiento en el espacio de mundo
            glm::vec3 currentPos = screenToWorld(x, y, cam->getViewAndProjectionMatrix(), width, height);

			translate(x - cam->getLastMouseX(), y - cam->getLastMouseY());

            cam->setLastMousePos(currentPos);
            cam->setLastMouseX(x);
            cam->setLastMouseY(y);
        }
        else if (currentMode == SCALE) {
            // Calcula el movimiento en el espacio de mundo
            glm::vec3 currentPos = screenToWorld(x, y, cam->getViewAndProjectionMatrix(), width, height);



            scale(x - cam->getLastMouseX(), y - cam->getLastMouseY());
            cam->setLastMousePos(currentPos);
            cam->setLastMouseX(x);
            cam->setLastMouseY(y);
        }
        else if (currentMode == EXTRUDE && selectedModelPosition != -1) {
			Model* selectedModel = &modelList[selectedModelPosition];
            if (selectedFacePosition == -1 && (*selectedModel).getFaces().size() > 0) {
                selectedFacePosition = 0;
				gizmoLookAtNormal();
            }
            
            if (!createdVertexExtruded) {
                gizmoLookAtNormal();
            }
            extrudeFace(x - cam->getLastMouseX(), y - cam->getLastMouseY());
            cam->setLastMouseX(x);
            cam->setLastMouseY(y);
		}
        
        // Si está en modo rotación y hay un modelo seleccionado, rota el modelo
        else if (currentMode == ROTATE && selectedModelPosition != -1) {
			Model* selectedModel = &modelList[selectedModelPosition];
			// Calcula la rotación entre las dos posiciones en la esfera
            glm::quat deltaQuat = quatFromVectors(lastPos, currentPos);
			// Rota todos los vértices del modelo alrededor de su centroide
            glm::vec3 center = centroidModel(*selectedModel);
            for (auto& v : selectedModel->getVertex()) {
                glm::vec3 localPos = v.pos - center;
                localPos = glm::vec3(deltaQuat * glm::vec4(localPos, 0.0f));
                v.pos = localPos + center;
            }
			calculateNormals(*selectedModel);
            updateModelMesh(*selectedModel, meshList[selectedModelPosition]);
            cam->setLastMousePos(currentPos);
        }
    }
    // Si se mantiene presionado el botón derecho, rota el gizmo
    if (cam->isRightButtonPressed() ) {
        if (showGizmo && selectedModelPosition != -1) {
            glm::quat deltaQuat = quatFromVectors(lastPos, currentPos);
            gizmo.rotateGizmo(deltaQuat);
            deleteMesh(gizmoMeshX);
            deleteMesh(gizmoMeshY);
            deleteMesh(gizmoMeshZ);
            gizmoMeshX = createMesh(gizmo.modelX);
            gizmoMeshY = createMesh(gizmo.modelY);
            gizmoMeshZ = createMesh(gizmo.modelZ);
            cam->setLastMousePos(currentPos);
        }
        // Si no está en modo rotación, rota la cámara
        else if (currentMode != ROTATE) {
            float sensitivity = 0.5f;
            glm::vec3 adjustedCurrentPos = lastPos + (currentPos - lastPos) * sensitivity;

            // Calcula la rotación entre las dos posiciones en la esfera
            glm::quat deltaQuat = quatFromVectors(lastPos, adjustedCurrentPos);
            glm::quat currentOrientation = cam->getOrientation();
            // Aplica la rotación a la orientación de la cámara
            glm::quat newOrientation = glm::normalize(deltaQuat * currentOrientation);
            cam->setOrientation(newOrientation);
            cam->setLastMousePos(currentPos);
        }
        
    }
    
}


// Dibuja un botón de modo en la interfaz ImGui.
void drawModeButton(const char* label, Mode expectedMode) {
    bool isActive = currentMode == expectedMode;
    if (isActive) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));
    }
    if (ImGui::Button(label)) {
        switch (expectedMode) {
        case CREATE_VERTEX:
            if (currentMode == CREATE_VERTEX) {
                resetSelection();
                currentMode = NONE_MODE;
                currentSelection = UNKNOWN;
                showGizmo = false;
            } else {
                resetSelection();
                currentMode = CREATE_VERTEX;
                currentSelection = FACE;
                showGizmo = false;
            }
            break;
        case ROTATE:
            if (currentMode == ROTATE) {
                resetSelection();
                currentMode = SELECT;
                currentSelection = UNKNOWN;
                showGizmo = false;
            } else {
                resetSelection();
                currentMode = ROTATE;
                currentSelection = MODEL;
                showGizmo = false;
            }
            break;
        case TRANSLATE:
            if (currentMode == TRANSLATE) {
                resetSelection();
                currentMode = SELECT;
                currentSelection = UNKNOWN;
                showGizmo = false;
                selectedAxis = NONE_AXIS;
            } else {
                currentMode = TRANSLATE;
                currentSelection = MODEL;
                updateGizmoForSelectedModel();
            }
            break;
        case SCALE:
            if (currentMode == SCALE) {
                resetSelection();
                currentMode = SELECT;
                currentSelection = UNKNOWN;
                showGizmo = false;
                selectedAxis = NONE_AXIS;
            }
            else {
                currentMode = SCALE;
                currentSelection = MODEL;
                updateGizmoForSelectedModel();
            }
            break;

        case EXTRUDE:
            if (currentMode == EXTRUDE) {
                resetSelection();
                currentMode = SELECT;
                currentSelection = UNKNOWN;
                showGizmo = false;
                selectedAxis = NONE_AXIS;
            }
            else {
				resetSelection();
                currentMode = EXTRUDE;
                currentSelection = FACE;
                
            }
            break;
        }
    }
    if (isActive) ImGui::PopStyleColor(3);
}

// Crea la ventana de control de ImGui con información y botones de modo.
static void createImGUIWindow() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::NewFrame();
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::Begin(u8"Control Panel");
    // Botón para crear cubo y seleccionarlo
    if (ImGui::Button(u8"Crear Cubo")) {
        size_t newModelIndex = createCubes();
        if (newModelIndex != -1) {
            // Seleccionar el nuevo modelo
            selectedModelPosition = newModelIndex;

            // Cambiar a modo traslación
            currentMode = TRANSLATE;
            currentSelection = MODEL;

            // Actualizar gizmo para el nuevo modelo
            updateGizmoForSelectedModel();
        }
    }
    ImGui::Separator();

    // Controles para crear prismas
    ImGui::Text(u8"Crear Módelo Básico:");
    static int sides = 6; // Número de lados por defecto

    ImGui::BeginGroup();
    ImGui::Text("Lados: %d", sides);

    ImGui::SameLine();
    if (ImGui::Button("-") && sides > 3) {
        sides--;
    }
    ImGui::SameLine();
    if (ImGui::Button("+") && sides < 32) {
        sides++;
    }
    ImGui::EndGroup();

    if (ImGui::Button(u8"Crear Prisma")) {
        size_t newModelIndex = createPrism(sides);
        if (newModelIndex != -1) {
            // Seleccionar el nuevo modelo
            selectedModelPosition = newModelIndex;

            // Cambiar a modo traslación
            currentMode = TRANSLATE;
            currentSelection = MODEL;

            // Actualizar gizmo para el nuevo modelo
            updateGizmoForSelectedModel();
        }
    }
    if (ImGui::Button(u8"Crear Pirámide")) {
		size_t newModelIndex = createPyramid(sides);
        if (newModelIndex != -1) {
            // Seleccionar el nuevo modelo
            selectedModelPosition = newModelIndex;

            // Cambiar a modo traslación
            currentMode = TRANSLATE;
            currentSelection = MODEL;

            // Actualizar gizmo para el nuevo modelo
            updateGizmoForSelectedModel();
        }
    }
    ImGui::Separator();
    if (ImGui::Button(u8"Borrar Seleccionado")) {
        if (selectedModelPosition != -1 && selectedModelPosition < modelList.size()) {
            // Eliminar el mesh de OpenGL
            deleteMesh(meshList[selectedModelPosition]);

            // Eliminar de las listas
            modelList.erase(modelList.begin() + selectedModelPosition);
            meshList.erase(meshList.begin() + selectedModelPosition);

            // Resetear selección
            resetSelection();
            showGizmo = false;
            selectedAxis = NONE_AXIS;
        }
    }
    ImGui::Separator();
    ImGui::Text(u8"Modo actual:");
    drawModeButton(u8"Crear Vértice", CREATE_VERTEX);
    drawModeButton(u8"Rotar Modelo", ROTATE);
    drawModeButton(u8"Trasladar Modelo/Vértice", TRANSLATE);
    drawModeButton(u8"Escalar Modelo", SCALE);
    drawModeButton(u8"Extruir Cara", EXTRUDE);

    ImGui::Separator();
    if (selectedModelPosition != -1) {
		Model* selectedModel = &modelList[selectedModelPosition];
        ImGui::Text(u8"Modelo seleccionado: %zu", selectedModelPosition);
        ImGui::Text(u8"Vértices: %zu", selectedModel->getVertex().size());
        ImGui::Text(u8"Caras: %zu", selectedModel->getFaces().size());
    }
    if (selectedVertexPosition != -1) {
        ImGui::Text(u8"Vértice seleccionado: %zu", selectedVertexPosition);
        ImGui::Text(u8"Posición: (%.2f, %.2f, %.2f)",
            selectedVertexWorldPos.x,
            selectedVertexWorldPos.y,
            selectedVertexWorldPos.z);
    }
	ImGui::Text(u8"Cámara Front (%.2f, %.2f, %.2f)", 
		cam->getFront().x, cam->getFront().y, cam->getFront().z);
	ImGui::Text(u8"Cámara Right (%.2f, %.2f, %.2f)", cam->getRight().x, cam->getRight().y, cam->getRight().z);
	ImGui::Text(u8"Cámara Up (%.2f, %.2f, %.2f)", cam->getUp().x, cam->getUp().y, cam->getUp().z);
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// Callback de renderizado principal. Dibuja la escena, los modelos y el gizmo.
static void renderSceneCB() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glm::vec3 lightPos(2.0f, 2.0f, 2.0f);
    glm::vec3 viewPos = cam->getPosition();

    // Obtener el programa de shaders actual 
    static GLint currentShaderProgram = 0;
    if (currentShaderProgram == 0) {
        glGetIntegerv(GL_CURRENT_PROGRAM, &currentShaderProgram);
    }
    GLint drawEdgesLocation = glGetUniformLocation(currentShaderProgram, "drawEdges");

    glUniform3f(lightPosLocation, lightPos.x, lightPos.y, lightPos.z);
    glUniform3f(viewPosLocation, viewPos.x, viewPos.y, viewPos.z);
    glm::mat4 mvp = cam->getViewAndProjectionMatrix();
    glUniformMatrix4fv(viewProjectionLocation, 1, GL_FALSE, &mvp[0][0]);

    // Dibujar relleno
    if (drawEdgesLocation != -1) {
        glUniform1i(drawEdgesLocation, 0);
    }
    for (size_t i = 0; i < meshList.size(); ++i) {
        glBindVertexArray(meshList[i].VAO);
        glDrawElements(GL_TRIANGLES, meshList[i].indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    // Dibujar contornos 
    if (drawEdgesLocation != -1) {
        glUniform1i(drawEdgesLocation, 1);
    }
    glLineWidth(2.0f); // Grosor de las líneas
    for (size_t i = 0; i < meshList.size(); ++i) {
        if (meshList[i].edgeVAO && meshList[i].edgeIndexCount > 0) {
            glBindVertexArray(meshList[i].edgeVAO);
            glDrawElements(GL_LINES, meshList[i].edgeIndexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
    glLineWidth(1.0f); // Restaurar grosor por defecto

    // Dibujar gizmo
    if (showGizmo) {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        if (drawEdgesLocation != -1) {
            glUniform1i(drawEdgesLocation, 0);
        }

        if (gizmoMeshX.VAO && gizmoMeshX.indexCount > 0) {
            glBindVertexArray(gizmoMeshX.VAO);
            glDrawElements(GL_TRIANGLES, gizmoMeshX.indexCount, GL_UNSIGNED_INT, 0);
        }
        if (gizmoMeshY.VAO && gizmoMeshY.indexCount > 0) {
            glBindVertexArray(gizmoMeshY.VAO);
            glDrawElements(GL_TRIANGLES, gizmoMeshY.indexCount, GL_UNSIGNED_INT, 0);
        }
        if (gizmoMeshZ.VAO && gizmoMeshZ.indexCount > 0) {
            glBindVertexArray(gizmoMeshZ.VAO);
            glDrawElements(GL_TRIANGLES, gizmoMeshZ.indexCount, GL_UNSIGNED_INT, 0);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    }

    createImGUIWindow();
    glutPostRedisplay();
    glutSwapBuffers();
}
// Callback para el cambio de tamaño de la ventana.
static void reshapeCB(int newWidth, int newHeight) {
    ImGui_ImplGLUT_ReshapeFunc(newWidth, newHeight);
    width = newWidth;
    height = newHeight;
    glViewport(0, 0, width, height);
    if (cam) {
        cam->updateProjection((float)width, (float)height);
    }
}


int main(int argc, char** argv)
{
    // verificar version de c++
    if (__cplusplus) {
        std::cout<<  __cplusplus << std::endl;
	}
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    int x = 200, y = 100;
    glutInitWindowPosition(x, y);
    glutCreateWindow("Editor 3D");
    // Inicialización de GLEW y OpenGL
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
        return 1;
    }
    // Inicialización de ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL3_Init("#version 330");
    // Configuración de OpenGL
    glClearColor(0.25f, 0.25f, 0.25f, 0.5f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);


    // Configuraciones adicionales importantes:
	glEnable(GL_BLEND); // Habilitar blending para transparencia
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POLYGON_OFFSET_LINE);
	glPolygonOffset(1.0f, 1.0f);

    // Para mejor calidad de líneas
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    // Carga de cubo inicial y shaders
    initCamara();
    createCubes();
    compileShaders(VERTEXSHADERFILENAME, FRAGMENTSHADERFILENAME, &viewProjectionLocation);
    glutReshapeFunc(reshapeCB);
    glutDisplayFunc(renderSceneCB);
    glutKeyboardFunc(keyboardCB);
    glutMouseFunc(mouseCB);
    glutMotionFunc(mouseMotionCB);
    glutMainLoop();
    // Limpieza de recursos
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();
    if (cam) delete cam;
    return 0;
}


