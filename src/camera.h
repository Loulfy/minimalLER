//
// Created by loulfy on 19/12/2022.
//

#ifndef MINIMALRT_CAMERA_H
#define MINIMALRT_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:

    glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(1);
    glm::vec3 right = glm::vec3(1);
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    // euler Angles
    float yaw = -90.f;
    float pitch = 0.f;
    // camera options
    float movementSpeed = 80.f;
    float mouseSensitivity = 0.1f;
    float zoom = 45.f;

    float lastX = 1200 / 2.0f;
    float lastY = 800 / 2.0f;
    bool firstMouse = true;
    bool lockMouse = false;

    Camera()
    {
        updateCameraVectors();
    }

    [[nodiscard]] glm::mat4 getViewMatrix() const
    {
        return glm::lookAt(position, position + front, up);
    }

    void keyboardCallback(int key, int action, float delta)
    {
        float velocity = movementSpeed * delta;
        if (key == 87) // Forward
            position += front * velocity;
        if (key == 83) // Backward
            position -= front * velocity;
        if (key == 65) // Left
            position -= right * velocity;
        if (key == 68) // Right
            position += right * velocity;
        if (key == 69) // Down
            position.y += 1 * velocity;
        if (key == 340) // Up
            position.y -= 1 * velocity;
    }

    void mouseCallback(double xposIn, double yposIn)
    {
        if(lockMouse)
            return;

        auto xpos = static_cast<float>(xposIn);
        auto ypos = static_cast<float>(yposIn);

        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

        lastX = xpos;
        lastY = ypos;

        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw   += xoffset;
        pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 tmp;
        tmp.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        tmp.y = sin(glm::radians(pitch));
        tmp.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(tmp);
        // also re-calculate the Right and Up vector
        right = glm::normalize(glm::cross(front, worldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        up = glm::normalize(glm::cross(right, front));
    }
};

#endif //MINIMALRT_CAMERA_H
