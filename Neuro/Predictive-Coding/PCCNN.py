import torch
import torch.nn as nn
import torch.nn.functional as F

MU = 0.70
NU = 1.0
ETA = 0.1
STEPS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
device

class PCInputLayer(nn.Module):
    def __init__(self, channels, next_channels, width, height, kernel=(3,3), stride=1, padding='same', bias=False, pool=1, forw_actv=nn.ReLU()):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.forw_actv = forw_actv
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, next_channels, kernel, stride=stride, padding=padding, bias=bias),
#             nn.Dropout(0.2),
#             nn.BatchNorm2d(next_channels),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=pool)
        
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.channels, self.width, self.height)).to(device)
        return e
        
    def pool(self, x):
        return self.pool(x)
    
    def step(self, x, td_pred):
        e = self.forw_actv(x-td_pred)
        return self.conv(e), e
    
    
class PCHiddenLayer(nn.Module):
    def __init__(self, prev_channels, channels, next_channels, width, height,  prev_kernel=(3,3), kernel=(3,3), prev_stride=1, stride=1, prev_padding=1, padding='same', bias=False, pool=1, upsample=1, back_actv=nn.Tanh(), forw_actv=nn.ReLU()):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.back_actv = back_actv
        self.forw_actv = forw_actv
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, next_channels, kernel, stride=stride, padding=padding, bias=bias),
#             nn.Dropout(0.2),
#             nn.BatchNorm2d(next_channels),
        )
        self.pool = nn.MaxPool2d(kernel_size=pool)
        self.upsample = nn.Upsample(scale_factor=upsample)
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(channels, prev_channels, prev_kernel, stride=prev_stride, padding=prev_padding, bias=bias),
#             nn.Dropout(0.2),
#             nn.BatchNorm2d(prev_channels),
        )
        
    def init_vars(self, batch_size):
        r = torch.zeros((batch_size, self.channels, self.width, self.height)).to(device)
        e = torch.zeros((batch_size, self.channels, self.width, self.height)).to(device)
        return r, e
    
    def pred(self, r):
        td_pred = self.back_actv(self.convT(r))
        td_pred = self.upsample(td_pred)
        return td_pred
        
    def step(self, bu_err, r, e, td_pred):
        r = NU*r + MU*bu_err - ETA*e
        e = self.forw_actv(r-td_pred)
        return self.pool(self.conv(e)), r, e
    
    
class PCFinalLayer(nn.Module):
    def __init__(self, prev_channels, channels, width, height, prev_kernel=(3,3), prev_stride=1, prev_padding=1, bias=False, upsample=1, back_actv=nn.Tanh()):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.back_actv = back_actv
        
        self.upsample = nn.Upsample(scale_factor=upsample)
        self.convT = nn.Sequential(
            nn.ConvTranspose2d(channels, prev_channels, prev_kernel, stride=prev_stride, padding=prev_padding, bias=bias),
#             nn.Dropout(0.2),
#             nn.BatchNorm2d(prev_channels),
        )
        
    def init_vars(self, batch_size):
        r = torch.zeros((batch_size, self.channels, self.width, self.height)).to(device)
        return r
    
    def pred(self, r):
        td_pred = self.back_actv(self.convT(r))
        td_pred = self.upsample(td_pred)
        return td_pred
        
    def step(self, bu_err, r, target=None):
        r = NU*r + MU*bu_err
        if target is not None:
            e = r - target
            r = r - ETA*e
        return r
    
    
class PCCNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.in_layer = PCInputLayer(input_channels, 64, 32, 32, 1, forw_actv=nn.ReLU())
        self.hid0_layer = PCHiddenLayer(input_channels, 64, 64, 32, 32, prev_padding=1, pool=2, forw_actv=nn.ReLU())
        
        self.hid1_layer = PCHiddenLayer(64, 64, 128, 16, 16, prev_padding=1, upsample=2, forw_actv=nn.ReLU())
        self.hid2_layer = PCHiddenLayer(64, 128, 128, 16, 16, prev_padding=1, pool=2, forw_actv=nn.ReLU())
        
        self.hid3_layer = PCHiddenLayer(128, 128, 256, 8, 8, prev_padding=1, upsample=2, forw_actv=nn.ReLU())
        self.hid4_layer = PCHiddenLayer(128, 256, 256, 8, 8, prev_padding=1, forw_actv=nn.ReLU())  
        self.hid5_layer = PCHiddenLayer(128, 256, 256, 8, 8, prev_padding=1, forw_actv=nn.ReLU())  
        self.fin_layer = PCFinalLayer(256, 256, 8, 8, prev_padding=1)
        
        self.fc_layer = nn.Linear(256*8*8, num_classes)
        
    def forward(self, x, target=None):
        batch_size = x.shape[0]
        in_e = self.in_layer.init_vars(batch_size)
        hid0_r, hid0_e = self.hid0_layer.init_vars(batch_size)
        hid1_r, hid1_e = self.hid1_layer.init_vars(batch_size)
        hid2_r, hid2_e = self.hid2_layer.init_vars(batch_size)
        hid3_r, hid3_e = self.hid3_layer.init_vars(batch_size)
        hid4_r, hid4_e = self.hid4_layer.init_vars(batch_size)
        hid5_r, hid5_e = self.hid4_layer.init_vars(batch_size)
        fin_r = self.fin_layer.init_vars(batch_size)
        
        for _ in range(STEPS):
            in_bu_e, in_e = self.in_layer.step(x, self.hid0_layer.pred(hid0_r))
            hid0_bu_e, hid0_r, hid0_e = self.hid0_layer.step(in_bu_e, hid0_r, hid0_e, self.hid1_layer.pred(hid1_r))
            hid1_bu_e, hid1_r, hid1_e = self.hid1_layer.step(hid0_bu_e, hid1_r, hid1_e, self.hid2_layer.pred(hid2_r))
            hid2_bu_e, hid2_r, hid2_e = self.hid2_layer.step(hid1_bu_e, hid2_r, hid2_e, self.hid3_layer.pred(hid3_r))
            hid3_bu_e, hid3_r, hid3_e = self.hid3_layer.step(hid2_bu_e, hid3_r, hid3_e, self.hid4_layer.pred(hid4_r))
            hid4_bu_e, hid4_r, hid4_e = self.hid4_layer.step(hid3_bu_e, hid4_r, hid4_e, self.fin_layer.pred(fin_r))
            fin_r = self.fin_layer.step(hid4_bu_e, fin_r, target)

        out = self.fc_layer(torch.flatten(F.relu(fin_r), start_dim=1))
        
        in_err = in_e.square().sum()/in_e.numel()
        hid0_err = hid0_e.abs().sum()/hid0_e.numel()
        hid1_err = hid1_e.abs().sum()/hid1_e.numel()
        hid2_err = hid2_e.abs().sum()/hid2_e.numel()
        hid3_err = hid3_e.abs().sum()/hid3_e.numel()
        hid4_err = hid4_e.abs().sum()/hid4_e.numel()
        emag = in_err + hid0_err + hid1_err + hid2_err + hid3_err + hid4_err
        
        return [out, emag]