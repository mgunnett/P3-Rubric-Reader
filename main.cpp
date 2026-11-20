// rubric_reader.cpp - bare-minimum MVP
//
// STEP 1 (once): rubric_reader calibrate sample.pdf
//   Shows page 1. Press R until it is upright, then ENTER.
//   Then drag a box around each option the console asks for (ENTER = save, C = skip).
//   Saves reference.png and boxes.txt.
//
// STEP 2:         rubric_reader sample.pdf
//   Writes results.csv: one row per page, plus a Flags column.

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace cv;

const int    DARK_PIXEL_LIMIT = 150;   // a pixel darker than this counts as ink (0 = black, 255 = white)
const double MARK_INK_AMOUNT  = 0.05;  // a marked box has 5% more ink than the emptiest option
const int    MIN_MATCHES      = 30;    // matching features needed to trust a page alignment
const int    MEMBER_COUNT     = 6;

vector<string> TOPICS = {"PresentationOfDesign", "QualityOfDesign", "PresentationDelivery", "QandA"};

// One box on the reference page, e.g. "QandA", 5, (x, y, width, height)
struct Box {
    string fieldName;
    int scoreValue;
    Rect area;
};

// Make every page grayscale with its long side 900 px, so all sizes match.
Mat preparePage(const Mat& colorImage) {
    Mat grayPage;
    cvtColor(colorImage, grayPage, COLOR_BGR2GRAY);
    double scale = 900.0 / max(grayPage.cols, grayPage.rows);
    resize(grayPage, grayPage, Size(), scale, scale);
    return grayPage;
}

// Turn each PDF page into an image using Poppler's pdftoppm.
vector<Mat> loadPdf(const string& pdfPath) {
    vector<Mat> pages;
    if (filesystem::path(pdfPath).extension() != ".pdf") return pages;  // wrong file type
    filesystem::remove_all("tmp_pages");
    filesystem::create_directory("tmp_pages");
    string command = "pdftoppm -r 150 -png \"" + pdfPath + "\" tmp_pages/p";
    if (system(command.c_str()) != 0) return pages;                     // pdftoppm failed
    vector<string> pageImageFiles;
    for (const auto& file : filesystem::directory_iterator("tmp_pages"))
        pageImageFiles.push_back(file.path().string());
    sort(pageImageFiles.begin(), pageImageFiles.end());                 // page order (p-01, p-02, ...)
    for (const auto& imageFile : pageImageFiles) pages.push_back(preparePage(imread(imageFile)));
    return pages;
}

// Line a page up with reference.png by matching printed features (words, lines).
// This also fixes sideways pages and tilt.
// Returns an empty image if it can't match (blank page or not a rubric).
Mat alignToReference(const Mat& page, const Mat& reference) {
    Ptr<ORB> featureFinder = ORB::create(5000);
    vector<KeyPoint> pageFeatures, referenceFeatures;
    Mat pageDescriptions, referenceDescriptions;
    featureFinder->detectAndCompute(page, noArray(), pageFeatures, pageDescriptions);
    featureFinder->detectAndCompute(reference, noArray(), referenceFeatures, referenceDescriptions);
    if (pageDescriptions.empty()) return Mat();

    // For each feature on the page, find the 2 most similar features on the reference.
    // Keep the match only if the closest one is clearly better than the second closest.
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> candidateMatches;
    matcher.knnMatch(pageDescriptions, referenceDescriptions, candidateMatches, 2);
    vector<Point2f> pagePositions, referencePositions;
    for (const auto& twoClosest : candidateMatches) {
        if (twoClosest.size() == 2 && twoClosest[0].distance < 0.75 * twoClosest[1].distance) {
            pagePositions.push_back(pageFeatures[twoClosest[0].queryIdx].pt);
            referencePositions.push_back(referenceFeatures[twoClosest[0].trainIdx].pt);
        }
    }
    if ((int)pagePositions.size() < MIN_MATCHES) return Mat();

    // Work out how to stretch/rotate the page so the matched points land on the reference.
    // "agreeingMatches" marks which matches fit that answer. Too few = not really the same form.
    Mat agreeingMatches;
    Mat pageToReference = findHomography(pagePositions, referencePositions, RANSAC, 5.0, agreeingMatches);
    if (pageToReference.empty() || countNonZero(agreeingMatches) < MIN_MATCHES) return Mat();

    Mat alignedPage;
    warpPerspective(page, alignedPage, pageToReference, reference.size(),
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
    return alignedPage;
}

// Fraction of dark pixels in a box (skipping 3 px at the edges so the printed border doesn't count).
double inkInBox(const Mat& page, Rect boxArea) {
    Rect innerArea(boxArea.x + 3, boxArea.y + 3, boxArea.width - 6, boxArea.height - 6);
    innerArea &= Rect(0, 0, page.cols, page.rows);
    if (innerArea.area() <= 0) return 0;
    Mat darkPixels = page(innerArea) < DARK_PIXEL_LIMIT;
    return (double)countNonZero(darkPixels) / innerArea.area();
}

// Look at every option box of one field and decide which one is marked.
// Returns the score ("" if none). Sets flagReason when unsure.
string readField(const Mat& page, const vector<Box>& boxes, const string& fieldName, string& flagReason) {
    vector<int> optionValues;
    vector<double> optionInk;
    for (const auto& box : boxes) {
        if (box.fieldName == fieldName) {
            optionValues.push_back(box.scoreValue);
            optionInk.push_back(inkInBox(page, box.area));
        }
    }
    if (optionValues.empty()) return "";

    // Printed numbers put the same ink in every box, so compare each box to the emptiest one.
    double lowestInk = *min_element(optionInk.begin(), optionInk.end());
    int mostInkIndex = max_element(optionInk.begin(), optionInk.end()) - optionInk.begin();
    int markedCount = 0;
    for (double boxInk : optionInk)
        if (boxInk - lowestInk >= MARK_INK_AMOUNT) markedCount++;

    if (markedCount == 0) {
        flagReason = "no mark found";
        return "";
    }
    if (markedCount > 1) {
        flagReason = "more than one box marked";
        return to_string(optionValues[mostInkIndex]) + "?";
    }
    return to_string(optionValues[mostInkIndex]);
}

vector<Box> loadBoxes() {
    vector<Box> boxes;
    ifstream boxFile("boxes.txt");
    Box box;
    while (boxFile >> box.fieldName >> box.scoreValue
                   >> box.area.x >> box.area.y >> box.area.width >> box.area.height)
        boxes.push_back(box);
    return boxes;
}

int calibrate(const string& pdfPath) {
    vector<Mat> pages = loadPdf(pdfPath);
    if (pages.empty()) { cout << "Could not read " << pdfPath << "\n"; return 1; }
    Mat reference = pages[0];

    cout << "Press R to rotate, ENTER when the page is upright.\n";
    while (true) {
        imshow("Calibrate", reference);
        int pressedKey = waitKey(0);
        if (pressedKey == 'r' || pressedKey == 'R') rotate(reference, reference, ROTATE_90_CLOCKWISE);
        if (pressedKey == 13) break;  // 13 = ENTER
    }
    imwrite("reference.png", reference);

    // List of every box to mark.
    vector<pair<string, int>> boxesToMark;
    for (const auto& topic : TOPICS)
        for (int score : {0, 5, 10}) boxesToMark.push_back({topic, score});
    for (int memberNumber = 1; memberNumber <= MEMBER_COUNT; memberNumber++)
        for (int score = 0; score <= 5; score++)
            boxesToMark.push_back({"Member" + to_string(memberNumber), score});

    ofstream boxFile("boxes.txt");
    for (const auto& [fieldName, score] : boxesToMark) {
        cout << "Drag a box around: " << fieldName << " = " << score << "   (ENTER = save, C = skip)\n";
        Rect selectedArea = selectROI("Calibrate", reference);
        if (selectedArea.area() > 0)
            boxFile << fieldName << " " << score << " " << selectedArea.x << " " << selectedArea.y << " "
                    << selectedArea.width << " " << selectedArea.height << "\n";
    }
    destroyAllWindows();
    cout << "Saved reference.png and boxes.txt\n";
    return 0;
}

int readSheets(const string& pdfPath) {
    Mat reference = imread("reference.png", IMREAD_GRAYSCALE);
    vector<Box> boxes = loadBoxes();
    if (reference.empty() || boxes.empty()) { cout << "Run calibrate first.\n"; return 1; }

    vector<Mat> pages = loadPdf(pdfPath);
    if (pages.empty()) { cout << "Could not read " << pdfPath << " (must be a .pdf)\n"; return 1; }

    ofstream csvFile("results.csv");
    if (!csvFile) { cout << "Can't write results.csv (close it in Excel first)\n"; return 1; }

    // Header row
    csvFile << "Page,";
    for (const auto& topic : TOPICS) csvFile << topic << ",";
    csvFile << "TopicTotal,";
    for (int memberNumber = 1; memberNumber <= MEMBER_COUNT; memberNumber++) csvFile << "Member" << memberNumber << ",";
    csvFile << "Flags\n";

    for (int pageIndex = 0; pageIndex < (int)pages.size(); pageIndex++) {
        int pageNumber = pageIndex + 1;
        string pageFlags;
        csvFile << pageNumber << ",";

        Mat alignedPage = alignToReference(pages[pageIndex], reference);
        if (alignedPage.empty()) {
            pageFlags = "blank page or not a rubric";
            int emptyColumns = TOPICS.size() + 1 + MEMBER_COUNT;  // topics + total + members
            for (int column = 0; column < emptyColumns; column++) csvFile << ",";
        } else {
            // The four topics
            int topicTotal = 0;
            bool totalIsValid = true;
            for (const auto& topic : TOPICS) {
                string flagReason;
                string score = readField(alignedPage, boxes, topic, flagReason);
                if (flagReason.empty() && !score.empty()) topicTotal += stoi(score);
                else totalIsValid = false;
                if (!flagReason.empty()) pageFlags += topic + ": " + flagReason + "; ";
                csvFile << score << ",";
            }
            csvFile << (totalIsValid ? to_string(topicTotal) : "") << ",";

            // Team members (an empty row just means no member there, so it isn't flagged)
            for (int memberNumber = 1; memberNumber <= MEMBER_COUNT; memberNumber++) {
                string memberField = "Member" + to_string(memberNumber);
                string flagReason;
                string score = readField(alignedPage, boxes, memberField, flagReason);
                if (flagReason == "more than one box marked") pageFlags += memberField + ": " + flagReason + "; ";
                csvFile << score << ",";
            }
        }
        csvFile << "\"" << pageFlags << "\"\n";
        cout << "Page " << pageNumber << ": " << (pageFlags.empty() ? "OK" : pageFlags) << "\n";
    }
    filesystem::remove_all("tmp_pages");
    cout << "Done. Open results.csv\n";
    return 0;
}

// argumentCount / arguments are what you type after the program name.
int main(int argumentCount, char** arguments) {
    try {
        if (argumentCount == 3 && string(arguments[1]) == "calibrate") return calibrate(arguments[2]);
        if (argumentCount == 2) return readSheets(arguments[1]);
        cout << "Usage:\n  rubric_reader calibrate sample.pdf\n  rubric_reader sample.pdf\n";
    } catch (const exception& error) {
        cout << "Error: " << error.what() << "\n";
    }
    return 1;
}