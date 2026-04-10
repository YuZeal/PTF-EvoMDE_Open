import torch
import numpy as np
import random, json

from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
import torch.distributed as dist

num_layers = [4, 4, 6, 6, 4, 1]
def is_tensor(obj):
    """
    检查对象是否是 PyTorch 张量。
    
    参数:
    obj (any): 要检查的对象
    
    返回:
    bool: 如果对象是 PyTorch 张量，则返回 True；否则返回 False
    """
    return isinstance(obj, torch.Tensor)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

class NetworkArchSearchProblem(Problem):

    def __init__(self, predictor_func, n_var, **kwargs):
        self.workflow = kwargs.pop('workflow')
        self.custom_eval_hook = kwargs.pop('custom_eval_hook')
        self.single_epochs = kwargs.pop('single_epochs')
        self.min_op = kwargs.pop('min_op')
        self.logger = kwargs.pop('logger')
        self.generation_id = 1
        super().__init__(n_var=25, n_obj=2, n_constr=0,  # 变量数，目标数，约束数
                         type_var=np.int32)
        self.xl = np.zeros(self.n_var)  # 变量上界
        self.xu = np.concatenate([np.array([5] + [6] * (length - 1)) for length in num_layers])  # 变量下界
        self.predictor_func = predictor_func

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)
        # 假设 predictor 的 predict 方法接受一个二维数组，每行一个架构编码，返回对应的性能精度
        f_err = []
        flops_list = []
        for _x in x:
            alpha_index = decode_arch(_x)
            # latency = calculate_total_time(_x)
            # print(f"Evaluating latency for {_x}: {latency}")

            performance, flops = self.predictor_func(self.workflow, self.custom_eval_hook, self.single_epochs, alpha_index, **kwargs)
            # performance, flops = [1.0], 2.0

            if self.min_op:
                f_err.append(performance[0])  # 使用silog来作为适应度值
            else:
                f_err.append(-performance[0])
            flops_list.append(flops)
            
            # print('f_err:', f_err)
            # print('flops:', flops)

        for i, (_x, err, flo) in enumerate(zip(x, f_err, flops_list)):
            f[i, 0] = err
            f[i, 1] = flo

        if is_main_process():
            gen = self.generation_id
            self.generation_id += 1

            f_err_np = np.array(f_err, dtype=np.float32)
            flops_np = np.array(flops_list, dtype=np.float32)

            self.logger.info(f"[Gen {gen}] silog: mean={f_err_np.mean():.2f},  min={f_err_np.min():.2f},  max={f_err_np.max():.2f}")
            self.logger.info(f"[Gen {gen}] flops: mean={flops_np.mean():.2f}G, min={flops_np.min():.2f}G, max={flops_np.max():.2f}G")

        out["F"] = f

def evolution(crossover_type, n_iter, population_size, logger, predictor_func, workflow, custom_eval_hook, single_epochs, min_op, **kwargs):

    # 初始化优化问题
    problem = NetworkArchSearchProblem(predictor_func, n_var=25,
                                       workflow=workflow,
                                       custom_eval_hook=custom_eval_hook,
                                       single_epochs=single_epochs,
                                       min_op=min_op,
                                       logger=logger)
    # 初始化多目标求解器
    crossover_operator = get_crossover(crossover_type, prob=0.9) if crossover_type != "None" else get_crossover("int_two_point", prob=0.0)

    method = get_algorithm(
        "nsga2", pop_size=population_size, sampling=get_sampling("int_random"),  # initialize with current nd archs
        crossover=crossover_operator,
        mutation=get_mutation("int_pm", eta=1.0),  # eta越大，变异越接近原值
        eliminate_duplicates=True)
    # 启动优化
    print('start mini')
    res = minimize(
        problem, method, termination=('n_gen', n_iter), save_history=True, verbose=False)

    # 打印历史
    for gen in res.history:
        pop_X = gen.pop.get("X")
        pop_F = gen.pop.get("F")
        logger.info(f"Generation: {gen.n_gen}")
        logger.info(f"Solutions:\n{pop_X}")
        logger.info(f"Objective Values:\n{pop_F}")

    ### result
    optimal_solutions = res.X.tolist()
    optimal_objective_values = res.F.tolist()

    logger.info("==="*20)
    logger.info("Optimal Code:")
    logger.info(optimal_solutions)

    return optimal_solutions, optimal_objective_values


def calculate_total_time(input_indices, file_path='./Latency_table_for_kitti_3090.json'):
    """
    根据输入的分支索引列表计算总时间。
    
    Args:
        input_indices (list): 每一层的分支索引列表。
        file_path (str): 本地 JSON 文件路径。

    Returns:
        float: 总时间（秒）。
    """
    try:
        # 读取 JSON 文件
        with open(file_path, "r") as f:
            data = json.load(f)

        total_time = 0.0
        for layer_idx, branch_idx in enumerate(input_indices):
            layer_key = f"layer_{layer_idx}"  # 构建层的键名
            branch_key = f"branch_{branch_idx}"  # 构建分支的键名
            
            # 检查键是否存在
            if layer_key in data and branch_key in data[layer_key]:
                mean_time = data[layer_key][branch_key]["mean"]
                total_time += mean_time
            elif branch_idx == 6:  # 需对应候选操作中的skip
                pass
            else:
                print(f"Warning: Missing data for {layer_key}, {branch_key}")
        
        return total_time
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON.")
        return None
    

def uniform_sample_code(num=1):
    ret = []
    redu_max = 5
    norm_max = 6
    for _ in range(num):
        ind = []
        for length in num_layers:
            for i in range(length):
                tmp = random.randint(0,redu_max) if i == 0 else random.randint(0,norm_max)
                ind.append(tmp)
        ret.append(ind)
    return ret

def encode_arch(nested_list):  # [torch.tensor([0], device='cuda:0'),...]  -> [0,2,3,...]
    return [item.item() for sublist in nested_list for item in sublist]

def decode_arch(flat_list):  # [0,2,3,...]  -> [torch.tensor([0], device='cuda:0'),...]
    # 固定长度列表
    index = 0
    nested_list = []
    for length in num_layers:
        nested_sublist = []
        for _ in range(length):
            nested_sublist.append(torch.tensor([flat_list[index]]).cuda())
            index += 1
        nested_list.append(nested_sublist)
    return nested_list

def balanced_flops_friendly_weights(n: int, skip_index: int = 6) -> np.ndarray:
    raw = np.array([1.0 / (1 + i)**2.0 for i in range(n)], dtype=np.float32)
    if skip_index < n:    
        raw[skip_index] = raw[1]  # 让 skip 权重与 index=1 一致
    raw[1] = raw[0]
    prob = raw / raw.sum()
    return prob

from typing import List, Tuple
def sample_candidate(code_space: List[int]) -> Tuple:
    return tuple(
        np.random.choice(range(i), p=balanced_flops_friendly_weights(i))
        for i in code_space
    )


# 示例嵌套列表
# nested_list = [
#     [torch.tensor([0], device='cuda:0'), torch.tensor([4], device='cuda:0'), torch.tensor([4], device='cuda:0'), torch.tensor([6], device='cuda:0')],
#     [torch.tensor([5], device='cuda:0'), torch.tensor([0], device='cuda:0'), torch.tensor([5], device='cuda:0'), torch.tensor([1], device='cuda:0')],
#     [torch.tensor([1], device='cuda:0'), torch.tensor([6], device='cuda:0'), torch.tensor([3], device='cuda:0'), torch.tensor([5], device='cuda:0'), torch.tensor([6], device='cuda:0'), torch.tensor([1], device='cuda:0')],
#     [torch.tensor([3], device='cuda:0'), torch.tensor([0], device='cuda:0'), torch.tensor([6], device='cuda:0'), torch.tensor([5], device='cuda:0'), torch.tensor([5], device='cuda:0'), torch.tensor([4], device='cuda:0')],
#     [torch.tensor([5], device='cuda:0'), torch.tensor([4], device='cuda:0'), torch.tensor([5], device='cuda:0'), torch.tensor([2], device='cuda:0')],
#     [torch.tensor([3], device='cuda:0')]
# ]

# # 调用 encode_arch 函数
# flat_list = encode_arch(nested_list)
# print(flat_list)

# # 调用 decode_arch 函数
# reconstructed_list = decode_arch(flat_list)
# print(reconstructed_list)
