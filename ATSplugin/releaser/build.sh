/opt/ts/bin/tsxs \
-I/opt/trafficserver-8.0.0/include \
-I/opt/trafficserver-8.0.0/include/ts \
-I/opt/trafficserver-8.0.0/include/tscpp \
-I/opt/pytorch/torch/lib/tmp_install/include \
-I/root/anaconda3/include \
-L/opt/trafficserver-8.0.0/lib \
-L/opt/ts/lib \
-L/opt/pytorch/torch/lib/tmp_install/lib \
-L/root/anaconda3/lib \
-ltorch -ltscppapi -lpthread -lmkl_rt -lmkl_avx2 -lmkl_def  -lmkl_core \
-o releaser.so \
-c plugin.cc common.cc headers.cc

/opt/ts/bin/tsxs -o releaser.so -i
cp ./releaser.so /opt/ts/libexec/trafficserver