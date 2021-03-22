import torch
import torch.nn as nn
import sys
sys.path.append("..")
from thumb_instance_norm import ThumbInstanceNorm


class CNN(nn.Module):
    def __init__(self, layer, matrixSize=32):
        super(CNN, self).__init__()
        if (layer == 'r31'):
            # 256x64x64
            self.convs = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 64, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, matrixSize, 3, 1, 1))
        elif (layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 128, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, matrixSize, 3, 1, 1))
        
        # 32x8x8
        self.fc = nn.Linear(matrixSize * matrixSize, matrixSize * matrixSize)
        # self.fc = nn.Linear(32*64,256*256)
    
    def forward(self, x):
        out = self.convs(x)
        # 32x8x8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        # 32x64
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        # 32x32
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MulLayer(ThumbInstanceNorm):
    def __init__(self, layer, matrixSize=32):
        super(MulLayer, self).__init__(affine=False)
        self.snet = CNN(layer, matrixSize)
        self.cnet = CNN(layer, matrixSize)
        self.matrixSize = matrixSize
        self.thumb_cov = None
        self.thumb_mean = None
        self.style_cov = None
        self.style_mean = None
        self.transmatrix = None

        if (layer == 'r41'):
            self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)
        elif (layer == 'r31'):
            self.compress = nn.Conv2d(256, matrixSize, 1, 1, 0)
            self.unzip = nn.Conv2d(matrixSize, 256, 1, 1, 0)
    
    def forward(self, cF, sF, trans=True):
        if self.collection:
            cb, cc, ch, cw = cF.size()
            cFF = cF.view(cb, cc, -1)
            cMean = torch.mean(cFF, dim=2, keepdim=True)
            cMean = cMean.unsqueeze(3)
            self.thumb_mean = cMean
        cF = cF - self.thumb_mean.expand_as(cF)
        if self.collection:
            sb, sc, sh, sw = sF.size()
            sFF = sF.view(sb, sc, -1)
            sMean = torch.mean(sFF, dim=2, keepdim=True)
            sMean = sMean.unsqueeze(3)
            sF = sF - sMean.expand_as(sF)
            self.style_mean = sMean

        compress_content = self.compress(cF)
        b, c, h, w = compress_content.size()
        compress_content = compress_content.view(b, c, -1)
        if (trans):
            if self.collection:
                cMatrix = self.cnet(cF)
                sMatrix = self.snet(sF)
                sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
                cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)
                transmatrix = torch.bmm(sMatrix, cMatrix)
                self.transmatrix = transmatrix
                
            transfeature = torch.bmm(self.transmatrix, compress_content).view(b, c, h, w)
            out = self.unzip(transfeature)
            out = out + self.style_mean.expand_as(out)
            return out
        else:
            out = self.unzip(compress_content.view(b, c, h, w))
            out = out + self.thumb_mean.expand_as(out)
            return out
    
    def forwardv2(self, tF, pF, sF, trans=True):
        
        tb, tc, th, tw = tF.size()
        tFF = tF.view(tb, tc, -1)
        tMean = torch.mean(tFF, dim=2, keepdim=True)
        tF = tF - tMean.unsqueeze(3).expand_as(tF)
        pF = pF - tMean.unsqueeze(3).expand_as(pF)
        
        sb, sc, sh, sw = sF.size()
        sFF = sF.view(sb, sc, -1)
        sMean = torch.mean(sFF, dim=2, keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(pF)
        sF = sF - sMean.expand_as(sF)
        
        compress_patch = self.compress(pF)
        compress_thumb = self.compress(tF)
        
        b, c, h, w = compress_thumb.size()
        compress_thumb = compress_thumb.view(b, c, -1)
        compress_patch = compress_patch.view(b, c, -1)
        
        if (trans):
            tMatrix = self.cnet(tF)
            sMatrix = self.snet(sF)
            
            sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
            tMatrix = tMatrix.view(tMatrix.size(0), self.matrixSize, self.matrixSize)
            transmatrix = torch.bmm(sMatrix, tMatrix)
            transfeature_thumb = torch.bmm(transmatrix, compress_thumb).view(b, c, h, w)
            transfeature_patch = torch.bmm(transmatrix, compress_patch).view(b, c, h, w)
            out_t = self.unzip(transfeature_thumb.view(b, c, h, w))
            out_t = out_t + sMeanC
            out_p = self.unzip(transfeature_patch.view(b, c, h, w))
            out_p = out_p + sMeanC
        else:
            out_t = self.unzip(compress_thumb.view(b, c, h, w))
            out_t = out_t + tMean
            out_p = self.unzip(compress_patch.view(b, c, h, w))
            out_p = out_p + tMean
        
        return out_t, out_p

