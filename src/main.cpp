#include "stereo_calibration.hpp"
#include "stereo_match.hpp"

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified;
    cv::CommandLineParser parser(argc, argv,
                                 "{w|5|}{h|7|}{s|0.5|}{nr||}{help||}{@input|"
                                 "stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help();
    showRectified = parser.has("nr");
    imagelistfn = parser.get<string>("@input");
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    auto squareSize = parser.get<float>("s");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if (!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }

    StereoCalib(imagelist, boardSize, squareSize, true, true, false);
    stereo_match();
    return 0;
}