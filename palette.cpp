/*!
   \file "palette.cpp"
   \brief "finding palette on pics which are given by path"
   \author "Mehmet BOZOKLU"
   \date "22"/"May"/"2024"
*/

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
namespace fs = filesystem;

struct {
  int n_clusters{6};
  int resize{120};
  int win_w{512};
  int win_h{512};
  int color_w{128};
  int color_h{139};
  int colors;
  string path{"../dataset/"};
  double threshold{0.99};
  bool vertical{1};
  bool reverse{1};
}init_values;


/*!
   \brief "Loading settings:
          n_clusters  : color count of the clusters
          resize      : reducing size of main image
          win_w       : image window width
          win_h       : image window height
          color_w     : palette color width
          color_h     : palette color height
          colors      : n_cluster - background color
          path        : images path
          threshold   : similarity of palette and pattern
          vertical    : rotation of the palette
          reverse     : color depth darker to lighter"
*/
void read_settings(void) {
  string setting;
  try{
      ifstream Settings("../settings.txt");

      getline(Settings, setting);
      init_values.n_clusters = stoi(setting);
      cout << "n_c\t: " << init_values.n_clusters << endl;

      getline(Settings, setting);
      init_values.resize = stoi(setting);
      cout << "rs\t: " << init_values.resize << endl;

      getline(Settings, setting);
      init_values.win_w = stoi(setting);
      cout << "win_w\t: " << init_values.win_w << endl;

      getline(Settings, setting);
      init_values.win_h = stoi(setting);
      cout << "win_h\t: " << init_values.win_h << endl;

      getline(Settings, setting);
      init_values.color_w = stoi(setting);
      cout << "color_w\t: " << init_values.color_w << endl;

      getline(Settings, setting);
      init_values.color_h = stoi(setting);
      cout << "color_h\t: " << init_values.color_h << endl;

      getline(Settings, setting);
      init_values.path = setting;
      cout << "path\t: " << init_values.path << endl;

      getline(Settings, setting);
      init_values.threshold = stod(setting);
      cout << "thr\t: " << init_values.threshold << endl;

      getline(Settings, setting);
      init_values.vertical = stoi(setting)==1;
      cout << "ver\t: " << init_values.vertical << endl;

      getline(Settings, setting);
      init_values.reverse = stoi(setting)==1;
      cout << "rev\t: " << init_values.reverse << endl;

      Settings.close();
  }
  catch(const exception&){
      cout << "Error reading the settings file!" << endl;
  }
}


/*!
   \brief "Orientation of palette window"
   \return palette window size
*/
Point setwin(void){
    Point win;
    if(init_values.vertical){
      win.x = init_values.color_w;
      win.y = init_values.color_h * init_values.colors;
    }else{
      win.y = init_values.color_w;
      win.x = init_values.color_h * init_values.colors;
    }

    return win;
}

/*!
   \brief "create palette orientation"
   \param "Param @ver vertical"
   \param "Param @rev reverse"
   \param "Param @centers mat"
   \return "return palette image"
*/
Mat create_palette(const bool& ver, const bool& rev, const Mat& centers) {
    vector<Mat> palette;
    Mat model;

    int x,y;

    if(ver){
      x = init_values.color_h;
      y = init_values.color_w;
    }else{
      y = init_values.color_h;
      x = init_values.color_w;
    }

    // Iterate through the clusters and create cv::Mat objects for each
    for (size_t cluster_idx = 0; cluster_idx < centers.rows-1; ++cluster_idx) {
        // Create a cv::Mat with the specified dimensions and fill it with the center values
        Mat cluster_mat(x, y, CV_8UC3,
            Scalar(centers.at<float>(cluster_idx,0),
                  centers.at<float>(cluster_idx,1),
                  centers.at<float>(cluster_idx,2)));
        palette.push_back(cluster_mat);
    }

    // Reverse the palette (if needed)
    if(rev)reverse(palette.begin(), palette.end());

    if(ver)vconcat(palette, model);
    else hconcat(palette, model);

    return model;
}


/*!
   \brief "labels count"
   \param "@labels labels"
   \return "return of counts of labels"
*/
vector<int> bincount(const vector<int>& labels) {
    int max_value = *max_element(labels.begin(), labels.end()) + 1;
    vector<int> counts(max_value, 0);

    for (int label : labels) {
        counts[label]++;
    }

    return counts;
}



/*!
   \brief "path can be given via command line" Usage:
   ./palette /your/image/files/path/
   or
   ./palette
*/
int main(int argc, char* argv[]) {
    read_settings();
    init_values.colors = init_values.n_clusters - 1;

    if (argc > 1) {
        cout << argc << " " << argv[1] << endl;
        init_values.path = argv[1];
    }


    Point win = setwin();


    for (const auto & entry : fs::directory_iterator(init_values.path)) {
        cout << entry.path() << endl;
        Mat image = imread(entry.path(), IMREAD_COLOR);
        if(image.empty()) {
            cout << "Could not read the image: " << entry.path() << endl;
            return 1;
        }

        Mat gb;
        GaussianBlur(image, gb, Size(19,19), 0, 0, BORDER_DEFAULT);

        // Reduce complexity by resizing
        Mat data;
        resize(gb, data, Size(init_values.resize, init_values.resize));
        data = data.reshape(1);
        data = data.reshape(3,init_values.resize*init_values.resize);

        // Convert to CV_32F if needed
        if (data.type() != CV_32F) {
          data.convertTo(data, CV_32F);
        }

        Mat labels, centers;
        double compactness = kmeans(data, init_values.n_clusters, labels,
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
                10, KMEANS_RANDOM_CENTERS, centers);

        cv::sort(centers, centers, SORT_EVERY_COLUMN | SORT_ASCENDING);
        vector<int> cluster_sizes = bincount(labels);

        Mat model;
        model = create_palette(init_values.vertical, init_values.reverse, centers);
        namedWindow("palette", WINDOW_NORMAL);
        resizeWindow("palette", win.x, win.y);
        imshow("palette", model);

        Mat res;
        // TM_CCOEFF_NORMED
        matchTemplate(gb, model, res, 5);

        // Find locations where the normalized correlation coefficient is above the threshold
        std::vector<Point> loc;
        for (int row = 0; row < res.rows; ++row) {
          for (int col = 0; col < res.cols; ++col) {
            if (res.at<float>(row, col) >= init_values.threshold) {
              loc.push_back(Point(col, row));
            }
          }
        }

        if(init_values.reverse)reverse(loc.begin(), loc.end());
        int unique, temp = 0;

        for (const Point& pt : loc) {
          if(init_values.vertical)unique=pt.x;
          else unique=pt.y;
          if (abs(unique - temp) > 7) {
            rectangle(image, Point(pt.x,pt.y), Point(pt.x + win.x, pt.y + win.y), Scalar(0, 0, 255), 2);
            std::string text = "Palette: " + std::to_string(pt.x + win.x) + ", " + std::to_string(pt.y + win.y);
            putText(image, text, Point(pt.x,pt.y+win.y+70) , FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0));
            temp = unique;
          }
        }

        if(temp==0)cout << "Palette not found!" << endl;

        namedWindow(entry.path(), WINDOW_NORMAL);
        resizeWindow(entry.path(), init_values.win_w, init_values.win_h);
        imshow(entry.path(), image);

        int k = waitKey(0); // Wait for a key in the window for next pic waitKey(0);
        destroyAllWindows();
    }

    destroyAllWindows();
    return 0;
}
