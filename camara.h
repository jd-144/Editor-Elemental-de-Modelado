#pragma once

#ifndef CAMARA_H
#define CAMARA_H

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camara {
private:
    glm::mat4 projection;
    glm::vec3 position;
    glm::quat orientation;

    // Variables para control de ratón
    bool leftButtonPressed;
    glm::vec3 lastMousePos;

	int lastMouseX;
	int lastMouseY;

	bool rightButtonPressed;

    float left;
    float right;
    float bottom; 
    float top;
    float zNear;
    float zFar;

public:
    Camara();

    Camara(const glm::vec3& pos, const glm::vec3& target, const glm::vec3& up,
       float left, float right, float bottom, float top, float zNear, float zFar);

    glm::mat4 getProjection() const;
    glm::vec3 getPosition() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getViewAndProjectionMatrix() const;
    void move(const glm::vec3& direction);

    bool isLeftButtonPressed() const { return leftButtonPressed; }
    void setLeftButtonPressed(bool pressed) { leftButtonPressed = pressed; }
	bool isRightButtonPressed() const { return rightButtonPressed; }
	void setRightButtonPressed(bool pressed) { rightButtonPressed = pressed; }
    glm::vec3 getLastMousePos() const { return lastMousePos; }
    void setLastMousePos(const glm::vec3& pos) { lastMousePos = pos; }
    glm::quat getOrientation() const { return orientation; }
    void setOrientation(const glm::quat& newOrientation) { orientation = newOrientation; }

    glm::vec3 getFront() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;

    void updateProjection(float width, float height) {
        if (height == 0.0f) return;

        float aspect = width / height;
		top = tanf(glm::radians(45.0f) / 2.0f) * zNear;
        right = top * aspect;
        left = -right;
		bottom = -top;
        

        projection = glm::frustum(left, right, bottom, top, zNear, zFar);
    }

	int getLastMouseX() { return lastMouseX; }
	int getLastMouseY() { return lastMouseY; }
	void setLastMouseX(int x) { lastMouseX = x; }
	void setLastMouseY(int y) { lastMouseY = y; }
};

#endif // CAMARA_H