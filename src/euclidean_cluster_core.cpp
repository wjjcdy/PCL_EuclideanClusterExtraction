#include "euclidean_cluster_core.h"

EuClusterCore::EuClusterCore(ros::NodeHandle &nh)
{

    seg_distance_ = {15, 30, 45, 60};                  // 将整个点云分割5份
    cluster_distance_ = {0.5, 1.0, 1.5, 2.0, 2.5};     // 云团聚集最远距离参数
    sub_point_cloud_ = nh.subscribe("/filtered_points_no_ground", 5, &EuClusterCore::point_cb, this);

    pub_bounding_boxs_ = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/detected_bounding_boxs", 5);

    ros::spin();
}

EuClusterCore::~EuClusterCore() {}

void EuClusterCore::publish_cloud(const ros::Publisher &in_publisher,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr,
                                  const std_msgs::Header &in_header)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header = in_header;
    in_publisher.publish(cloud_msg);
}

void EuClusterCore::voxel_grid_filer(pcl::PointCloud<pcl::PointXYZ>::Ptr in, pcl::PointCloud<pcl::PointXYZ>::Ptr out, double leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(in);
    filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    filter.filter(*out);
}

void EuClusterCore::cluster_segment(pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc,
                                    double in_max_cluster_distance, std::vector<Detected_Obj> &obj_list)
{

    // 创建kd tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    //create 2d pc， 创建一个备份
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*in_pc, *cloud_2d);
    //make it flat，将其进行投影至地平面，即忽略高度信息
    for (size_t i = 0; i < cloud_2d->points.size(); i++)
    {
        cloud_2d->points[i].z = 0;
    }

    //将点云构建kd tree结构
    if (cloud_2d->points.size() > 0)
        tree->setInputCloud(cloud_2d);

    std::vector<pcl::PointIndices> local_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclid;   //欧几里德聚类
    euclid.setInputCloud(cloud_2d);                          
    euclid.setClusterTolerance(in_max_cluster_distance);     // 临近搜索距离，太小，分的太多，太大则容易将独立的分成一个
    euclid.setMinClusterSize(MIN_CLUSTER_SIZE);              // 设置点云簇最小个数size
    euclid.setMaxClusterSize(MAX_CLUSTER_SIZE);              // 
    euclid.setSearchMethod(tree);                            // 搜索结构为kdtree，即搜索机制
    euclid.extract(local_indices);                           // 输出点云簇

    for (size_t i = 0; i < local_indices.size(); i++)        // 遍历点云簇
    {
        // the structure to save one detected object
        Detected_Obj obj_info;

        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();

        for (auto pit = local_indices[i].indices.begin(); pit != local_indices[i].indices.end(); ++pit)
        {
            //fill new colored cluster point by point        // 获取点云簇中x,y,z中最小最大，即边界值
            pcl::PointXYZ p;
            p.x = in_pc->points[*pit].x;
            p.y = in_pc->points[*pit].y;
            p.z = in_pc->points[*pit].z;

            obj_info.centroid_.x += p.x;
            obj_info.centroid_.y += p.y;
            obj_info.centroid_.z += p.z;

            if (p.x < min_x)
                min_x = p.x;
            if (p.y < min_y)
                min_y = p.y;
            if (p.z < min_z)
                min_z = p.z;
            if (p.x > max_x)
                max_x = p.x;
            if (p.y > max_y)
                max_y = p.y;
            if (p.z > max_z)
                max_z = p.z;
        }

        //min, max points
        obj_info.min_point_.x = min_x;
        obj_info.min_point_.y = min_y;
        obj_info.min_point_.z = min_z;

        obj_info.max_point_.x = max_x;
        obj_info.max_point_.y = max_y;
        obj_info.max_point_.z = max_z;

        //calculate centroid, average
        if (local_indices[i].indices.size() > 0)             // 求出点云簇形心
        {
            obj_info.centroid_.x /= local_indices[i].indices.size();
            obj_info.centroid_.y /= local_indices[i].indices.size();
            obj_info.centroid_.z /= local_indices[i].indices.size();
        }

        //calculate bounding box                             // 计算点云簇长 宽 高
        double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
        double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
        double height_ = obj_info.max_point_.z - obj_info.min_point_.z;

        obj_info.bounding_box_.header = point_cloud_header_;

        obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
        obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
        obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

        obj_info.bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
        obj_info.bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
        obj_info.bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

        obj_list.push_back(obj_info);
    }
}

// 实现欧几里得聚类
void EuClusterCore::cluster_by_distance(pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc, std::vector<Detected_Obj> &obj_list)
{
    //cluster the pointcloud according to the distance of the points using different thresholds (not only one for the entire pc)
    //in this way, the points farther in the pc will also be clustered
    // 聚类的距离根据点到雷达中心的距离，进行变化。 越远可聚类的距离也就越远
    //0 => 0-15m d=0.5
    //1 => 15-30 d=1
    //2 => 30-45 d=1.6
    //3 => 45-60 d=2.1
    //4 => >60   d=2.6

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_pc_array(5);
    
    // 开辟5个pcl点云指针空间
    for (size_t i = 0; i < segment_pc_array.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        segment_pc_array[i] = tmp;
    }

    // 根据距离将所有点云分成5份
    for (size_t i = 0; i < in_pc->points.size(); i++)
    {
        pcl::PointXYZ current_point;
        current_point.x = in_pc->points[i].x;
        current_point.y = in_pc->points[i].y;
        current_point.z = in_pc->points[i].z;

        // 计算欧几里得距离
        float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

        // 如果点的距离大于120m, 忽略该点
        if (origin_distance >= 120)
        {
            continue;
        }

        if (origin_distance < seg_distance_[0])
        {
            segment_pc_array[0]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[1])
        {
            segment_pc_array[1]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[2])
        {
            segment_pc_array[2]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[3])
        {
            segment_pc_array[3]->points.push_back(current_point);
        }
        else
        {
            segment_pc_array[4]->points.push_back(current_point);
        }
    }

    std::vector<pcl::PointIndices> final_indices;
    std::vector<pcl::PointIndices> tmp_indices;
    // 将5个部分点云分别进行点云分割
    for (size_t i = 0; i < segment_pc_array.size(); i++)
    {
        cluster_segment(segment_pc_array[i], cluster_distance_[i], obj_list);
    }
}

// 接收点云进行处理
void EuClusterCore::point_cb(const sensor_msgs::PointCloud2ConstPtr &in_cloud_ptr)
{
    //定义两个pcl点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    point_cloud_header_ = in_cloud_ptr->header;
    
    // 将ros msg 转换为pcl点云格式
    pcl::fromROSMsg(*in_cloud_ptr, *current_pc_ptr);
    // down sampling the point cloud before cluster， 降采样
    voxel_grid_filer(current_pc_ptr, filtered_pc_ptr, LEAF_SIZE);

    std::vector<Detected_Obj> global_obj_list;               // 定位物体列表
    cluster_by_distance(filtered_pc_ptr, global_obj_list);   // 类聚功能实现

    jsk_recognition_msgs::BoundingBoxArray bbox_array;       // box矩阵

    for (size_t i = 0; i < global_obj_list.size(); i++)
    {
        bbox_array.boxes.push_back(global_obj_list[i].bounding_box_);
    }
    bbox_array.header = point_cloud_header_;

    pub_bounding_boxs_.publish(bbox_array);
}