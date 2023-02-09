import torch.nn as nn
import torch


class ThumbInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True):
        super(ThumbInstanceNorm, self).__init__()
        self.thumb_mean = None
        self.thumb_std = None
        self.collection = True
        if affine == True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1, 1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True))

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, x, thumb=None):
        if self.training:
            thumb_mean, thumb_std = self.calc_mean_std(thumb)
            x = (x - thumb_mean) / thumb_std * self.weight + self.bias
            thumb = (thumb - thumb_mean) / thumb_std * self.weight + self.bias
            return x, thumb
        else:
            if self.collection:
                thumb_mean, thumb_std = self.calc_mean_std(x)
                self.thumb_mean = thumb_mean
                self.thumb_std = thumb_std
            x = (x - self.thumb_mean) / self.thumb_std * self.weight + self.bias
            return x
        

class ThumbAdaptiveInstanceNorm(ThumbInstanceNorm):
    def __init__(self):
        super(ThumbAdaptiveInstanceNorm, self).__init__(affine=False)
    
    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        if self.collection == True:
            thumb_mean, thumb_std = self.calc_mean_std(content_feat)
            # thumb_mean = torch.rand(thumb_mean.shape).cuda()
            # thumb_std = torch.rand(thumb_std.shape).cuda()
            # print(thumb_mean.shape, thumb_std.shape)
            self.thumb_mean = thumb_mean
            self.thumb_std = thumb_std
    
        normalized_feat = (content_feat - self.thumb_mean.expand(
            size)) / self.thumb_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ThumbWhitenColorTransform(ThumbInstanceNorm):
    def __init__(self):
        super(ThumbWhitenColorTransform, self).__init__(affine=False)
        self.thumb_mean = None
        self.style_mean = None
        self.trans_matrix = None

        
    def forward(self, cF, sF, wct_mode):
        if self.collection:
            cFSize = cF.size()
            c_mean = torch.mean(cF, 1)  # c x (h x w)
            c_mean = c_mean.unsqueeze(1)
            self.thumb_mean = c_mean
            
        cF = cF - self.thumb_mean
    
        if self.collection:
            if wct_mode == 'cpu':
                contentCov = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double()
            else:
                contentCov = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().cuda()
            c_u, c_e, c_v = torch.svd(contentCov, some=False)
        
            k_c = cFSize[0]
            for i in range(cFSize[0]):
                if c_e[i] < 0.00001:
                    k_c = i
                    break
        
            c_d = (c_e[0:k_c]).pow(-0.5)
            step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
            step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
            thumb_cov = step2
    
            sFSize = sF.size()
            s_mean = torch.mean(sF, 1)
            self.style_mean = s_mean.unsqueeze(1)
            sF = sF - s_mean.unsqueeze(1).expand_as(sF)
            styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
            s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
            k_s = sFSize[0]
            for i in range(sFSize[0]):
                if s_e[i] < 0.00001:
                    k_s = i
                    break
            s_d = (s_e[0:k_s]).pow(0.5)
            style_cov = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t()))
            self.trans_matrix = torch.mm(style_cov, thumb_cov)

        targetFeature = torch.mm(self.trans_matrix, cF)
        targetFeature = targetFeature + self.style_mean.expand_as(targetFeature)
        return targetFeature



def init_thumbnail_instance_norm(model, collection):
    for name, layer in model.named_modules():
        if isinstance(layer, ThumbInstanceNorm):
            layer.collection = collection