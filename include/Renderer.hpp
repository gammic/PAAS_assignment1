
//
// The original author of the rendering code is Aaron Brown (https://github.com/awbrown90).
// His code has been slightly modified to make it more structured.
//

#ifndef LIDAR_OBSTACLE_DETECTION_RENDERER_HPP
#define LIDAR_OBSTACLE_DETECTION_RENDERER_HPP


#include "../include/Box.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>
#include <string>


namespace lidar_obstacle_detection
{

  struct Color
  {

    float r, g, b;

    Color(float setR, float setG, float setB)
        : r(setR), g(setG), b(setB)
    {}
  };

  static bool lidarActivated;

  enum class CameraAngle
  {
    XY, TopDown, Side, FPS
  };

  class Renderer
  {
  private:
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    unsigned long long rays_counter_;

  public:

    static void setLidarStatus();

    static bool getLidarStatus();

    Renderer();

    static void keyboardCallback(const pcl::visualization::KeyboardEvent &event);

    void removeShape(int id);

    void RenderHighway();

    void RenderRays(const Eigen::Vector3f& origin, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    void addCircle(float centroid_x, float centroid_y, int id);
    
    void ClearRays();

    void ClearViewer();

    void addText(float centroid_x, float centroid_y, float centroid_z,const std::string& id);

    void RenderPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          const std::string& name,
                          const Color& color = Color(1,1,1));

    void RenderPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                          const std::string& name,
                          const Color& color = Color(-1,-1,-1));

    void RenderBox(const Box& box, int id, const Color& color = Color(1,0,0), float opacity = 1.0);

    void RenderBox(const BoxQ& box, int id, const Color& color = Color(1,0,0), float opacity = 1.0);

    void InitCamera(CameraAngle view_angle);

    bool WasViewerStopped() const;

    void SpinViewerOnce() const;

  };
}

#endif