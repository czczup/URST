import torch.nn as nn
import torch.nn.functional as F
import torch
from style_subnet import StyleSubnet
from enhance_subnet import EnhanceSubnet
from refine_subnet import RefineSubnet


class MT(nn.Module):
    def __init__(self):
        super().__init__()

        self.style_subnet = StyleSubnet()
        self.enhance_subnet = EnhanceSubnet()
        self.refine_subnet = RefineSubnet()

    def replaceResIN2d(self, net, key):
        net[f'{key}.in1.weight'] = net.pop(
            f'{key}.in1.weight').reshape(1, -1, 1, 1)
        net[f'{key}.in1.bias'] = net.pop(
            f'{key}.in1.bias').reshape(1, -1, 1, 1)
        net[f'{key}.in2.weight'] = net.pop(
            f'{key}.in2.weight').reshape(1, -1, 1, 1)
        net[f'{key}.in2.bias'] = net.pop(
            f'{key}.in2.bias').reshape(1, -1, 1, 1)
        return net

    def load(self, model):
        style_subnet = torch.load(
            'models/style_subnet_' + model + '.pt', map_location='cpu').state_dict()
        style_subnet['in4.weight'] = style_subnet.pop(
            'in4.scale').reshape(1, -1, 1, 1)
        style_subnet['in4.bias'] = style_subnet.pop(
            'in4.shift').reshape(1, -1, 1, 1)
        style_subnet['in5.weight'] = style_subnet.pop(
            'in5.scale').reshape(1, -1, 1, 1)
        style_subnet['in5.bias'] = style_subnet.pop(
            'in5.shift').reshape(1, -1, 1, 1)
        for i in range(4, 7):
            style_subnet = self.replaceResIN2d(style_subnet, f'res{i}')
        for i in range(1, 4):
            style_subnet[f'rgb_in{i}.weight'] = style_subnet.pop(
                f'rgb_in{i}.scale').reshape(1, -1, 1, 1)
            style_subnet[f'rgb_in{i}.bias'] = style_subnet.pop(
                f'rgb_in{i}.shift').reshape(1, -1, 1, 1)
            style_subnet[f'l_in{i}.weight'] = style_subnet.pop(
                f'l_in{i}.scale').reshape(1, -1, 1, 1)
            style_subnet[f'l_in{i}.bias'] = style_subnet.pop(
                f'l_in{i}.shift').reshape(1, -1, 1, 1)
            style_subnet = self.replaceResIN2d(style_subnet, f'rgb_res{i}')
            style_subnet = self.replaceResIN2d(style_subnet, f'l_res{i}')
        self.style_subnet.load_state_dict(style_subnet)

        enhance_subnet = torch.load(
            'models/enhance_subnet_' + model + '.pt', map_location='cpu').state_dict()
        for i in range(1, 7):
            enhance_subnet = self.replaceResIN2d(enhance_subnet, f'res{i}')
        for i in range(1, 8):
            enhance_subnet[f'in{i}.weight'] = enhance_subnet.pop(
                f'in{i}.weight').reshape(1, -1, 1, 1)
            enhance_subnet[f'in{i}.bias'] = enhance_subnet.pop(
                f'in{i}.bias').reshape(1, -1, 1, 1)
        self.enhance_subnet.load_state_dict(enhance_subnet)

        refine_subnet = torch.load(
            'models/refine_subnet_' + model + '.pt', map_location='cpu').state_dict()
        for i in range(1, 4):
            refine_subnet = self.replaceResIN2d(refine_subnet, f'res{i}')
        for i in range(1, 6):
            refine_subnet[f'in{i}.weight'] = refine_subnet.pop(
                f'in{i}.weight').reshape(1, -1, 1, 1)
            refine_subnet[f'in{i}.bias'] = refine_subnet.pop(
                f'in{i}.bias').reshape(1, -1, 1, 1)
        self.refine_subnet.load_state_dict(refine_subnet)

        # for error 'conv2d has no attribute padding_mode'
        # for m in self.modules():
        #     if 'Conv' in str(type(m)):
        #         setattr(m, 'padding_mode', 'none')

    def forward(self, x):
        # x = F.interpolate(x, 256, mode='bilinear', align_corners=True)
        y1, _ = self.style_subnet(x)
        y2, _ = self.enhance_subnet(y1)
        y3, _ = self.refine_subnet(y2)

        return y3
