#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <iomanip>
#include <cmath>
#include <string>
#include <set>
#include <fstream>
using namespace std;

#define A_L 6
#define INPUT_L 9

template<typename T>
void PrintArray(const T input_array[], const unsigned int array_size)
{
    for (int i = 0; i < array_size; ++i)
    {
        std::cout << "[" << i << "]:" << input_array[i]<<" ";
    }
    std::cout<<std::endl;
}


int choose_action_lower(int choosableList[], int reqBI){
    for(int i=reqBI; i>=0; i--){
       if(choosableList[i] == 1)
           return i;
    }
    return 5;
}


int choose_action_closest(int choosableList[], int reqBI){
    int min_dist = 10;
    int action = -1;
    for(int i=0; i<=A_L-1; i++){
        int dist = abs(reqBI - i);
        if(choosableList[i] == 1 && min_dist > dist){
            action = i;
            min_dist = dist;
        }
    }
    if(action == -1){
        return 5;
    }
    else{
        return action;
    }
}


int choose_action_highest(int choosableList[]){
    for(int i=A_L-2; i>=0; i--){
        if(choosableList[i] == 1){
            return i;
        }
    }
    return 5;
}


int main(int argc, const char* argv[]) {
    if (argc != INPUT_L+1) { // bitrate  laBitrate  buffer  throughput  downloadT  reqBI  videoName  busy
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    enum policy{no_policy, RL, lower, closest, highest};
    policy p = lower;


    double throughput_mean = 2297514.2311790097;    // bps
    double throughput_std = 4369117.906444455;      // bps

    int BITRATES[] = {350, 600, 1000, 2000, 3000};

    double bitrate;         // kbps
    double lastBitrate = 0; // kbps
    double buffer = 0;      // ms
    double hThroughput = 0; // bps
    double mThroughput = 0; // bps
    int reqBI = 0; // 0, 1, 2, 3, 4
    string videoName = "";
    string method = "";

    for(int i=1; i<=INPUT_L; i++){
        if(i==1){                       // bitrate
            bitrate =       atof(argv[i]) / BITRATES[4];
        }else if(i==2){                 // lastBitrate
            lastBitrate =   atof(argv[i]) / BITRATES[4];
        }else if(i==3){                 // buffer
            buffer =        (atof(argv[i]) / 1000 - 30) / 10;
        }else if(i==4){                 // hit throughput
            hThroughput =   (atof(argv[i]) - throughput_mean)/throughput_std;
        }else if(i==5){                 // miss throughput
            mThroughput =   (atof(argv[i]) - throughput_mean)/throughput_std;
        }else if(i==6){                 // reqBI
            reqBI =         atoi(argv[i]);
        }else if(i==7){                 // videoName
            videoName =     argv[i];
        }else if(i==8){
            method =        argv[i];
        }else if(i==9){
            busy =          atof(argv[i])
        }
    }
    double data[7] = {bitrate, lastBitrate, buffer, hThroughput, mThroughput, busy};

    int choosable[6] = {1,1,1,1,1,1};
    set<string> cachedSet;
    string line;
    ifstream in("/opt/RL/build/cachedFile_20190612_pure.txt"); // videoName version chunk_count
    while(getline(in, line))
    {
        cachedSet.insert(line.substr(0, line.find_last_of(" ")));
    }
    in.close();
    for(int i=0; i<5; i++){
        if(cachedSet.find(videoName+" "+to_string(i+1))==cachedSet.end()){
            choosable[i] = 0;
        }
    }
    int action=reqBI;
    int sum = 0;
    for(int i = 0; i < A_L-1; i++){
        sum += choosable[i];
    }
    sum = 1;
    if(sum == 0)
    {
        action = 5;
    }
    else
    {
        if(method == "no_policy")
        {
                action = reqBI;
        }
        else if(method == "RL")
        {
            std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("/opt/RL/build/model.pt");
            assert(module != nullptr);
            std::vector<torch::jit::IValue> inputs;
            torch::Tensor s = torch::from_blob(data, {1, 5});
            inputs.push_back(s);
            at::Tensor output = module->forward(inputs).toTensor();
            auto output_accessor = output.accessor<float, 2>(); //1*5
            float output_float[A_L] = {0};
            for(int i = 0; i < output_accessor.size(1); i++){
                output_float[i] = output_accessor[0][i];
            }
            cout<<"data:";
            PrintArray(data,5);
            cout<<"output_float:";
            PrintArray(output_float, A_L);
            cout<<"choosable:";
            PrintArray(choosable, A_L);
            float max = -999999;
            int index = 0;
            for(int i = 0; i < A_L; i++) {
                if (output_float[i] > max and choosable[i]==1){
                    max=output_float[i];
                    index = i;
                }
            }
            action = index;
            ofstream out("/opt/RL/build/cachedFile_20190612_pure_rl.txt",ios::app);
            for(int i = 0; i < A_L-1; i++){
                out<<videoName+"_"+to_string(i+1)+" "+to_string(output_float[i])<<endl;
            }
            out.close();
        }
        else if(method == "lower")
        {
            action = choose_action_lower(choosable, reqBI);
        }
        else if(method == "closest")
        {
            action = choose_action_closest(choosable, reqBI);
        }
        else if(method == "highest")
        {
            action = choose_action_highest(choosable);
        }
        else
        {
            action = reqBI;
        }
    }

    if(reqBI == action){
        action = 5;
    }
    printf("%d", action);

}