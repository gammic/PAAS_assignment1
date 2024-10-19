#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include "../include/Renderer.hpp"
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <chrono>
#include <unordered_set>
#include "../include/tree_utilities.hpp"

#define USE_PCL_LIBRARY
using namespace lidar_obstacle_detection;
typedef std::unordered_set<int> my_visited_set_t;

void setupKdtree(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, int dimension)
{
    for (int i = 0; i < cloud->size(); ++i)
    {
        tree->insert({cloud->at(i).x, cloud->at(i).y, cloud->at(i).z}, i);
    }
}

void proximity(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int target_ndx, my_pcl::KdTree* tree, float distanceTol, my_visited_set_t& visited, std::vector<int>& cluster, int max)
{
	if (cluster.size() < max)
    {
        cluster.push_back(target_ndx);
        visited.insert(target_ndx);
        std::vector<float> point {cloud->at(target_ndx).x, cloud->at(target_ndx).y, cloud->at(target_ndx).z};
        std::vector<int> neighborNdxs = tree->search(point, distanceTol);
        for (int neighborNdx : neighborNdxs)
        {
            if (visited.find(neighborNdx) == visited.end())
            {
                proximity(cloud, neighborNdx, tree, distanceTol, visited, cluster, max);
            }
            if (cluster.size() >= max)
            {
                return;
            }
        }
    }
}

std::vector<pcl::PointIndices> euclideanCluster(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, float distanceTol, int setMinClusterSize, int setMaxClusterSize)
{
	my_visited_set_t visited{};                                           
	std::vector<pcl::PointIndices> clusters;
    std::vector<int> cluster;                                             
    for (int c=0; c<cloud->size(); ++c){
        if (visited.find(c)==visited.end()){
            cluster.clear();
            proximity(cloud, c, tree, distanceTol,  visited, cluster, setMaxClusterSize);
            if (cluster.size()>=setMinClusterSize){
                pcl::PointIndices clusterIndices;
                clusterIndices.indices=cluster;
                clusters.push_back(clusterIndices);
            }
        }
    }
	return clusters;	
}

void ProcessAndRenderPointCloud (Renderer& renderer, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PointXYZ source_point(0,0,0);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.1f, 0.1f, 0.1f); 
    sor.filter(*cloud_filtered);
    pcl::CropBox<pcl::PointXYZ> cb(true);
    cb.setInputCloud(cloud_filtered);
    cb.setMin(Eigen::Vector4f (-20, -6, -2, 1));
    cb.setMax(Eigen::Vector4f ( 30, 7, 5, 1));
    cb.filter(*cloud_filtered); 
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    int i = 0, nr_points = (int) cloud_filtered->size ();
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.25); 
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ()); 
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ()); 
    while (cloud_filtered->size () > 0.5 * nr_points){
            seg.setInputCloud (cloud_filtered);
            seg.segment (*inliers, *coefficients); 
            if (inliers->indices.size () == 0){
                std::cerr << "Stima modello planare impossibile per il dataset fornito." << std::endl;
                break;
            }
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud (cloud_filtered); 
            extract.setIndices (inliers);
            extract.setNegative (false); 
            extract.filter (*cloud_segmented);
            extract.setNegative (true); 
            extract.filter(*cloud_plane);
            cloud_filtered.swap(cloud_plane);
        }
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered); 
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    #ifdef USE_PCL_LIBRARY
        ec.setClusterTolerance (0.28); 

        ec.setMinClusterSize (100);
        ec.setMaxClusterSize (5000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered);
        
        ec.extract(cluster_indices);
    #else
        my_pcl::KdTree treeM;
        treeM.set_dimension(3);
        setupKdtree(cloud_filtered, &treeM, 3);
        cluster_indices = euclideanCluster(cloud_filtered, &treeM, 0.28, 100, 5000);
    #endif
    std::vector<Color> colors = {Color(1,0,0), Color(1,1,0), Color(0,0,1), Color(1,0,1), Color(0,1,1)};
    int CID = 0;
    int index = 0;
    renderer.RenderPointCloud(cloud_segmented,"originalCloud"+std::to_string(CID),colors[4]);
    for (std::vector<pcl::PointIndices>::const_iterator c = cluster_indices.begin (); c != cluster_indices.end (); ++c)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator p_c = c->indices.begin (); p_c != c->indices.end (); ++p_c){
            cloud_cluster->push_back ((*cloud_filtered)[*p_c]); }
        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);
        Box box{minPt.x, minPt.y, minPt.z,
        maxPt.x, maxPt.y, maxPt.z};
        pcl::PointXYZ center((minPt.x + maxPt.x)/2.0f,
                                (minPt.y + maxPt.y)/2.0f,
                                (minPt.z + maxPt.z)/2.0f);
        float dist = std::sqrt(std::pow(source_point.x - center.x, 2) + 
                                    std::pow(source_point.y - center.y, 2) +
                                    std::pow(source_point.z - center.z, 2));
        renderer.addText(center.x, center.y, center.z+10, std::to_string(dist));
        if(dist<=5 && center.x>source_point.x){
            renderer.RenderBox(box, index, colors[0]);
        }else{
            renderer.RenderBox(box, index, colors[2]);
        }
        ++CID;
        index++;
    }  

}

int main(int argc, char* argv[])
{
    if(argc!=2){
        std::cerr << "Nessuna cartella di cloud stream specificata, uscita dal programma in corso" << std::endl;
        return 1;
    }
    Renderer renderer;
    renderer.InitCamera(CameraAngle::XY);
    renderer.ClearViewer();
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<boost::filesystem::path> stream(boost::filesystem::directory_iterator{argv[1]},
                                                boost::filesystem::directory_iterator{});
    std::sort(stream.begin(), stream.end());
    auto streamIterator = stream.begin();
    while (not renderer.WasViewerStopped())
    {
        renderer.ClearViewer();
        pcl::PCDReader reader;
        reader.read (streamIterator->string(), *input_cloud);
        auto startTime = std::chrono::steady_clock::now();
        ProcessAndRenderPointCloud(renderer,input_cloud);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "[PointCloudProcessor<PointT>::ReadPcdFile] Loaded "
        << input_cloud->points.size() << " data points from " << streamIterator->string() <<  "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
        streamIterator++;
        if(streamIterator == stream.end())
            streamIterator = stream.begin();
        renderer.SpinViewerOnce();
    }
}
