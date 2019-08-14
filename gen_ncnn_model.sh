cp weights/export.onnx /home/neptune/github/ncnn/build/tools/onnx

cd /home/neptune/github/ncnn/build/tools/onnx

./onnx2ncnn export.onnx yolov3.param yolov3.bin

../ncnn2mem yolov3.param yolov3.bin yolov3.id.h yolov3.mem.hbash

cp yolov3.param yolov3.bin /home/neptune/cppprj/ncnn_tst/cmake-build-debug/


