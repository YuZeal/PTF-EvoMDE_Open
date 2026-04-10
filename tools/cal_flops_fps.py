import torch
import numpy as np

# 计算FLOPs 和 Params     # TODO
'''https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file'''
from calflops import calculate_flops
from torchvision import models
import sys
def cal_flops_fps(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # input_shape = (1, 3, 640, 480)
    input_shape = (1, 3, 352, 1120)
    print(input_shape)
    with open('./tmp_FLOPs.log', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        # model = models.resnet50() 
        flops, macs, params = calculate_flops(model=model, 
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=2)
        sys.stdout = original_stdout
        print("net FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))        
        #Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M 

    ### compute time,fps
    '''code from https://zhuanlan.zhihu.com/p/376925457'''
    dummy_input = torch.randn(1, 3, 352, 1120,dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings=np.zeros((repetitions,1))
    print("begin to test time, fps!")
    #GPU-WARM-UP
    for _ in range(100):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            torch.cuda.synchronize()
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms   Std@5 {std_syn:.3f}ms   FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    assert False,"test over"