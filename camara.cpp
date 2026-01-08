#include "camara.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cmath>

Camara::Camara()
    : position(0.0f, 0.0f, 0.0f),
    orientation(1.0f, 0.0f, 0.0f, 0.0f),
    projection(1.0f),
    leftButtonPressed(false),
    rightButtonPressed(false),
    lastMousePos(0.0f),
    lastMouseX(0),
    lastMouseY(0),
	left(-1.0f),
	right(1.0f),
	bottom(-1.0f),
	top(1.0f),
	zNear(0.1f),
	zFar(100.0f)
{
}

Camara::Camara(const glm::vec3& pos, const glm::vec3& target, const glm::vec3& up,
    float leftP, float rightP, float bottomP, float topP, float zNearP, float zFarP)
{
    position = pos;
	left = leftP;
	right = rightP;
	bottom = bottomP;
	top = topP;
	zNear = zNearP;
	zFar = zFarP;

    glm::vec3 forward = glm::normalize(target - pos);
    glm::vec3 rightVec = glm::normalize(glm::cross(forward, up));
    glm::vec3 correctedUp = glm::cross(rightVec, forward);

    glm::mat4 lookAtMat = glm::lookAt(pos, target, correctedUp);
    orientation = glm::quat_cast(glm::inverse(lookAtMat));
    orientation = glm::normalize(orientation);

    projection = glm::frustum(left, right, bottom, top, zNear, zFar);

    leftButtonPressed = false;
    rightButtonPressed = false;
    lastMousePos = glm::vec3(0.0f);
    lastMouseX = lastMouseY = 0;
}

glm::mat4 Camara::getProjection() const {
    return projection;
}

glm::vec3 Camara::getPosition() const {
    return position;
}

glm::mat4 Camara::getViewMatrix() const {
    // La vista es la inversa de la transformación de la cámara en el mundo
	glm::mat4 rotation = glm::mat4_cast(glm::conjugate(orientation));
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), -position);
    return rotation * translation;
}

glm::mat4 Camara::getViewAndProjectionMatrix() const {
    return projection * getViewMatrix();
}

void Camara::move(const glm::vec3& direction) {
    // Mueve en el sistema local de la cámara
    glm::vec3 localMove = orientation * direction;
    position += localMove;
}

glm::vec3 Camara::getFront() const {
    return glm::normalize(orientation * glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::vec3 Camara::getRight() const {
    return glm::normalize(orientation * glm::vec3(1.0f, 0.0f, 0.0f));
}

glm::vec3 Camara::getUp() const {
    return glm::normalize(orientation * glm::vec3(0.0f, 1.0f, 0.0f));
}
