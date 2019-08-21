python -m onnxsim weights/export.onnx weights/export_sim.onnx

cp weights/export_sim.onnx /home/neptune/github/ncnn/build/tools/onnx

cd /home/neptune/github/ncnn/build/tools/onnx

./onnx2ncnn export_sim.onnx yolov3.param yolov3.bin

../ncnn2mem yolov3.param yolov3.bin yolov3.id.h yolov3.mem.hbash

cp yolov3.param yolov3.bin /home/neptune/github/momandai/ncnn_momandai/cmake-build-debug/examples


