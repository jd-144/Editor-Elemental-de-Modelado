#pragma once
#include <string>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <GL/glew.h>
#include <vector>
#include <array>
#include <set>


struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 color;
	

	Vertex() {}
	Vertex(float x, float y, float z, float r, float g, float b) {
		pos = glm::vec3(x, y, z);
		color = glm::vec3(r, g, b);
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	}
	Vertex(float x, float y, float z) {
		pos = glm::vec3(x, y, z);
		// El color ahora se define en el shader, pero mantenemos el atributo
		color = glm::vec3(0.6f, 0.7f, 0.8f); // Gris azulado por defecto
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	}

	
};


struct MeshBuffers {
	GLuint VBO;
	GLuint EBO;
	GLuint VAO;
	GLuint indexCount;

	GLuint edgeEBO;
	GLuint edgeVAO;
	GLuint edgeIndexCount;

	MeshBuffers() {
		VBO = 0;
		EBO = 0;
		VAO = 0;
		indexCount = 0;

		edgeEBO = 0;
		edgeVAO = 0;
		edgeIndexCount = 0;
	}
};

struct Face {
    std::vector<std::array<unsigned int, 3>> index;
    Face() {}
    Face(std::initializer_list<unsigned int> ind) {
        if (ind.size() % 3 != 0) {
            throw std::invalid_argument("The number of indices must be a multiple of 3.");
        }
        std::vector<unsigned int> tmp(ind);
        for (size_t i = 0; i < tmp.size(); i += 3) {
            index.push_back(std::array<unsigned int, 3>{ tmp[i], tmp[i + 1], tmp[i + 2] });
        }
    }
    Face(const std::vector<std::array<unsigned int, 3>>& triangles) {
        index = triangles;
    }
    Face(const std::vector<unsigned int>& flatTriangles) {
        if (flatTriangles.size() % 3 != 0) {
            throw std::invalid_argument("The number of indices must be a multiple of 3.");
        }
        for (size_t i = 0; i < flatTriangles.size(); i += 3) {
            index.push_back(std::array<unsigned int, 3>{ flatTriangles[i], flatTriangles[i + 1], flatTriangles[i + 2] });
        }
    }
};

class Model {
	private:
	std::vector<Face> faces;
	std::vector<Vertex> vertex;

    public:
	Model() {}
	Model(std::vector<Vertex>& vert, std::vector<Face>& fcs)
		: vertex(vert), faces(fcs) {
	}
	std::vector<Vertex>& getVertex() {
		return vertex;
	}
	std::vector<Face>& getFaces() {
		return faces;
	}
	void addVertex(const Vertex& v) {
		vertex.push_back(v);
	}

	

	std::vector<unsigned int> getAllIndex() {
		std::vector<unsigned int> allIndex;
		for (const auto& face : faces) {
			for (const auto& triangle : face.index) {
				allIndex.push_back(triangle[0]);
				allIndex.push_back(triangle[1]);
				allIndex.push_back(triangle[2]);
			}
		}
		return allIndex;
	}

};
glm::vec3 createArrow(Model& arrowModel, const glm::vec3& position, const glm::vec3& color, const glm::vec3& direction, const glm::quat& rotation);
void calculateNormals(Model& model);

class Gizmo {
public:
    glm::vec3 position; // centro de la base inferior de las flechas del guizmo
    glm::quat rotation;
    Model modelX;
    Model modelY;
    Model modelZ;
	glm::vec3 currentNormal;

    Gizmo() : position(0.0f), rotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {}

    Gizmo(const glm::vec3& pos) : position(pos), rotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
        updateGizmoModels();
    }

    void setPosition(const glm::vec3& pos) {
        position = pos;
        updateGizmoModels();
    }

    void setRotation(const glm::quat& newRotation) {
        rotation = newRotation;
        updateGizmoModels();
    }

    void rotateGizmo(const glm::quat& deltaRotation) {
        rotation = deltaRotation * rotation;
        updateGizmoModels();
    }
	void setNormal(const glm::vec3& normal) { currentNormal = normal; }
	glm::vec3 getCurrentNormal() const { return currentNormal; }

private:
    void updateGizmoModels() {
        createArrow(modelX, position, glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(1.0f, 0.0f, 0.0f), rotation);
        createArrow(modelY, position, glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f), rotation);
        createArrow(modelZ, position, glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f), rotation);

		
    }

    
};

using namespace std;
bool ReadFile(const char* fileName, string& destino);
glm::quat quatFromVectors(const glm::vec3& from, const glm::vec3& to);
glm::vec3 screenToWorld(int x, int y, const glm::mat4& viewProjectionMatrix, int screenWidth, int screenHeight);
glm::vec3* rayIntersectsFace(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, Model& model, const Face& face, unsigned int& triangleIndex, float& outT);
void subdivideFace(Model& model, Face& face, const glm::vec3& newVertexPos, unsigned int triangleIndex);
void updateModelMesh(Model& model, MeshBuffers& mesh);
bool vertexExists(const std::vector<Vertex>& vertices, const glm::vec3& position, float epsilon = 0.001f);
bool rayIntersectsVertex(const glm::vec3& rayOrigin, const glm::vec3& rayDir, Vertex vertex, float epsilon, float& outDistance);

void compileShaders(const char* vertexShaderFileName, const char* fragmentShaderFileName, GLint* viewProjectionLocationPtr);
void addShader(GLuint shaderProgram, const char* shaderText, GLenum shaderType);
glm::vec3 calculateFaceNormal(Model& model, const Face& face);

// Devuelve una lista de pares de índices que representan los bordes únicos de una cara.
std::vector<std::pair<unsigned int, unsigned int>> perimeterIndex(Face& face);

glm::vec3 centroidModel(Model& model);
glm::vec3 centroidFace(Model& model, Face& face);
