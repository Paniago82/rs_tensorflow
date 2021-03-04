#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include "../include/cppflow/include/cppflow/cppflow.h"

#include <ros/package.h>

using namespace uima;


class TenserFlowAnnotator : public Annotator
{
private:
    std::string modelName;

public:
    std::string rosPath;
    std::string fullPath;
    std::string picturePath;
    cv::Mat color;
    TyErrorId initialize(AnnotatorContext &ctx)
    {
        outInfo("initialize");
        rosPath = ros::package::getPath("rs_tensorflow");
        fullPath = rosPath + "/data/1"; //"/data/EASE_R02_1obj_test" ;
        picturePath = rosPath + "/data/pictures/";
        return UIMA_ERR_NONE;
    }

    TyErrorId destroy()
    {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
    {
        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
        cas.get(VIEW_CLOUD,*cloud_ptr);
        cas.get(VIEW_COLOR_IMAGE_HD, color);

        cv::Rect roi;
        roi.width = 250;
        roi.height = 250;
        roi.x = 250;
        roi.y = 330;

        std::string imagePath = picturePath + "pipe.jpg";
        outInfo("-------------------------1-------------------------");
        bool check = cv::imwrite(imagePath, cv::Mat(color, roi));

        if (check == false) {
            outError("Mission - Saving the image, FAILED");
        }

        outInfo("-------------------------2-------------------------");
        auto input = cppflow::decode_jpeg(cppflow::read_file(imagePath));
        // Cast it to float, normalize to range [0, 1], and add batch_dimension
        outInfo("-------------------------3-------------------------");
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        //input = input / 255.f;
        input = cppflow::expand_dims(input, 0);
        outInfo("-------------------------4-------------------------");
        cppflow::model model(fullPath);
        outInfo("-------------------------5-------------------------");
        auto output = model({{"serving_default_rescaling_1_input:0", input}},{"StatefulPartitionedCall:0"});
        outInfo("-------------------------6-------------------------");
        std::cout << output[0].dtype() << std::endl;
        std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;
        outInfo("-------------------------7-------------------------");



        return UIMA_ERR_NONE;
    }
};

/*
//, picturePath + "000409-color.jpg", picturePath + "000437-color.jpg", picturePath + "000467-color.jpg", picturePath + "000568-color.jpg"
std::string img_paths[] {picturePath + "000335-color.jpg"};

cppflow::model model(fullPath);


cppflow::tensor cm {1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0};
cppflow::tensor cm_shape {1,3,3};

cm = cppflow::reshape(cppflow::cast(cm, TF_DOUBLE, TF_FLOAT), cm_shape);

outInfo("Input cameramatrix: " << std::endl << cm);

for (auto img_path : img_paths)
{
    outInfo("" << img_path << ":");

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(img_path)));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);

    auto output = model({{"serving_default_camera_matrix_input:0", cm}, {"serving_default_input_2:0", input}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"})[0]; // this ([0]) takes only the pose

    outInfo("output : " << output);
}

/*
//---------------------------------CODE---------------------------------
Model model("../../cppflow/examples/coco/frozen_inference_graph.pb");
Tensor outNames1{model, "num_detections"};
Tensor outNames2{model, "detection_scores"};
Tensor outNames3{model, "detection_boxes"};
Tensor outNames4{model, "detection_classes"};

Tensor inpName{model, "image_tensor"};

// Read image
cv::Mat img, inp;
img = cv::imread("../../cppflow/examples/coco/test.jpg", cv::IMREAD_COLOR);

int rows = img.rows;
int cols = img.cols;

cv::resize(img, inp, cv::Size(300, 300));
cv::cvtColor(inp, inp, cv::COLOR_BGR2RGB);

// Put image in Tensor
std::vector<uint8_t > img_data;
img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
inpName.set_data(img_data, {1, 300, 300, 3});

model.run(inpName, {&outNames1, &outNames2, &outNames3, &outNames4});

// Visualize detected bounding boxes.
int num_detections = (int)outNames1.get_data<float>()[0];
for (int i=0; i<num_detections; i++) {
    int classId = (int)outNames4.get_data<float>()[i];
    float score = outNames2.get_data<float>()[i];
    auto bbox_data = outNames3.get_data<float>();
    std::vector<float> bbox = {bbox_data[i*4], bbox_data[i*4+1], bbox_data[i*4+2], bbox_data[i*4+3]};
    if (score > 0.3) {
        float x = bbox[1] * cols;
        float y = bbox[0] * rows;
        float right = bbox[3] * cols;
        float bottom = bbox[2] * rows;

        cv::rectangle(img, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
    }
}

cv::imshow("Image", img);
cv::waitKey(0);
*/

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(TenserFlowAnnotator)