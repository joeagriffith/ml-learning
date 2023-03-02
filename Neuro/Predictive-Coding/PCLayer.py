import torch
import torch.nn as nn

class PCLayer(nn.Module):
    def __init__(self, 
                 e_shape,
                 r_shape,
                 kernel,
                 
                 nu,
                 mu,
                 eta,

                 maxpool=1,
                 forw_actv=nn.ReLU(),
                 td_actv=nn.Tanh(),
                 
                 padding=0,
                 **kwargs,
                ):
        super().__init__()
        self.e_shape = e_shape
        self.r_shape = r_shape

        self.nu = nu
        self.mu = mu
        self.eta = eta

        self.device = "cpu"

        self.conv = nn.Sequential(
            nn.Conv2d(e_shape[0], r_shape[0], kernel, padding=padding, **kwargs),
            nn.MaxPool2d(kernel_size=maxpool)
        )
        self.forw_actv = forw_actv
        
        self.convT = nn.Sequential(
            nn.Upsample(scale_factor=maxpool),
            nn.ConvTranspose2d(r_shape[0], e_shape[0], kernel, padding=padding, **kwargs),
            td_actv,
        )

        self.rec_conv = nn.Sequential(
            nn.Conv2d(r_shape[0], r_shape[0], (10,10), padding="same")
        )
    
    # def to(self, device):
    #     self.conv = self.conv.to(device)
    #     self.convT = self.convT.to(device)
    #     self.rec_conv = self.rec_conv.to(device)
    #     self.device = device
    
    def init_vars(self, batch_size):
        e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2])).to(self.device)
        r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2])).to(self.device)
        return e,r
    
    def forward(self, x, e, r, td_err=None):
        e = self.forw_actv(x - self.convT(r))
        # r = self.nu*self.rec_conv(r) + self.mu*self.conv(e)
        r = self.nu*r + self.mu*self.conv(e)
        if td_err is not None:
            r -= self.eta*td_err
        return e, r
