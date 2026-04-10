import logging
from collections import Iterable


class ArchGenerate_FNA(object):
    def __init__(self, super_network):
        self.primitives_normal = super_network.primitives_normal
        self.primitives_reduce = super_network.primitives_reduce
        if hasattr(super_network, 'primitives_for_head'):
            self.primitives_for_head = super_network.primitives_for_head
        self.num_layers = super_network.num_layers
        self.chs = super_network.search_params.net_scale.chs

    def update_arch_code(self, alphas):
        self.alphas = alphas
    
    def derive_archs(self, alphas, logger=None):  # 根据最大alphas权重选择架构模块 -> 改为根据alphas直接选择
        # flat = lambda t: [x for sub in t for x in flat(sub)] if isinstance(t, Iterable) else [t]
        self.update_arch_code(alphas)
        def _parse(codes):
            assert len(alphas) == sum(self.num_layers)

            final_stages = []
            final_stages.append(['k3_e1'])
            count = 0
            for num_layer in self.num_layers:
                stage = []
                for ii in range(num_layer):
                    code = codes[count]
                    if ii == 0:
                        op = self.primitives_reduce[code]
                    else:
                        op = self.primitives_normal[code]
                    stage.append(op)
                    count += 1
                final_stages.append(stage)  # stage = ['k3_e3', 'k3_e6', 'skip', 'skip']

            final_code = []
            for i, stage in enumerate(final_stages):
                if i in [1,2,3,5]:
                    stride = 2
                else:
                    stride = 1
                final_code.append([[self.chs[i], self.chs[i+1]], stage, stride])
            return ('|\n'.join(map(str, final_code)))
            """[[32, 16], ['k3_e1'], 1]|
            [[16, 24], ['k3_e3', 'k3_e6', 'skip', 'skip'], 2]|
            [[24, 32], ['k5_e3', 'k3_e3', 'k5_e3', 'k7_e3'], 2]|
            [[32, 64], ['k3_e6', 'k3_e6', 'k5_e6', 'k3_e6', 'k7_e6', 'k7_e6'], 2]|
            [[64, 96], ['k7_e6', 'k3_e6', 'k7_e3', 'k5_e6', 'k5_e6', 'skip'], 1]|
            [[96, 160], ['k7_e3', 'k5_e3', 'k3_e6', 'k5_e6'], 2]|
            [[160, 320], ['k3_e6'], 1]
            """

        net_config = _parse(alphas)
        logging.debug(net_config)
        return net_config