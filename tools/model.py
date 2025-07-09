import torch
import torch.nn as nn
import torch.distributions as tdist

class SocialCellLocal(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12):
        super(SocialCellLocal, self).__init__()

        # Spatial Section
        self.feat = nn.Conv1d(spatial_input,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        self.feat_act = nn.ReLU()
        self.highway_input = nn.Conv1d(spatial_input,
                                       spatial_output,
                                       1,
                                       padding=0)

        # Temporal Section
        self.highway = nn.Conv1d(temporal_input, temporal_output, 1, padding=0)
        self.tpcnn = nn.Conv1d(temporal_input,
                               temporal_output,
                               3,
                               padding=1,
                               padding_mode='zeros')

    def forward(self, v):
        v_shape = v.shape
        # Spatial Section
        v = v.permute(0, 3, 1, 2).reshape(v_shape[0] * v_shape[3], 
                                           v_shape[1], 
                                           v_shape[2])
        v_res = self.highway_input(v)
        v = self.feat_act(self.feat(v)) + v_res

        # Temporal Section
        v = v.permute(0, 2, 1)
        v_res = self.highway(v)
        v = self.tpcnn(v) + v_res

        # Final Output
        v = v.permute(0, 2, 1).reshape(v_shape[0], v_shape[3], 
                                        v_shape[1], 12).permute(0, 2, 3, 1)
        return v

class SocialCellGlobal(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12,
                 noise_weight=None):
        super(SocialCellGlobal, self).__init__()

        # Spatial Section
        self.feat = nn.Conv2d(spatial_input,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        self.feat_act = nn.ReLU()
        self.highway_input = nn.Conv2d(spatial_input,
                                       spatial_output,
                                       1,
                                       padding=0)
        # Temporal Section
        self.highway = nn.Conv2d(temporal_input, temporal_output, 1, padding=0)
        self.tpcnn = nn.Conv2d(temporal_input,
                               temporal_output,
                               3,
                               padding=1,
                               padding_mode='zeros')

        # Learnable weights
        self.noise_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_weight = noise_weight
        self.global_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.local_w = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Local Stream
        self.ped = SocialCellLocal(spatial_input=spatial_input,
                                   spatial_output=spatial_output,
                                   temporal_input=temporal_input,
                                   temporal_output=temporal_output)

    def forward(self, v, weight_select=1):
        # Spatial Section
        v_ped = self.ped(v)
        v_res = self.highway_input(v)
        v = self.feat_act(self.feat(v)) + v_res

        # Temporal Section
        v = v.permute(0, 2, 1, 3)
        v_res = self.highway(v)
        v = self.tpcnn(v) + v_res

        # Fuse Local and Global Streams
        v = v.permute(0, 2, 1, 3)
        v = self.global_w * v + self.local_w * v_ped
        return v

class SocialImplicit(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=2,
                 temporal_input=8,
                 temporal_output=12,
                 bins=[0, 0.01, 0.1, 1.2],
                 noise_weight=[0.05, 1, 4, 8]):
        super(SocialImplicit, self).__init__()
        
        # Store temporal parameters as attributes
        self.temporal_input = temporal_input
        self.temporal_output = temporal_output
        
        # Register buffers
        self.register_buffer('bins', torch.tensor(bins))
        self.register_buffer('noise_weight_tensor', torch.tensor(noise_weight))
        
        self.implicit_cells = nn.ModuleList([
            SocialCellGlobal(
                spatial_input=spatial_input,
                spatial_output=spatial_output,
                temporal_input=temporal_input,
                temporal_output=temporal_output,
                noise_weight=self.noise_weight_tensor
            ) for _ in range(len(bins))
        ])

    def forward(self, v, obs_traj, KSTEPS=20):
        device = v.device
        dtype = v.dtype
        
        # Generate noise with correct dimensions
        noise = torch.randn(KSTEPS, 1, 2, 1, 1, device=device, dtype=dtype)
        
        # Social-Zones Assignment
        norm = torch.linalg.norm(v[0, :, 0].t(), float('inf'), dim=1)
        displacement_idx = torch.bucketize(norm, self.bins, right=True) - 1
        
        # Initialize output tensor
        v_out = torch.zeros(KSTEPS, 2, self.temporal_output, v.shape[-1], 
                            device=device, dtype=dtype)
        
        # Process each social zone
        for i in range(len(self.bins)):
            select = displacement_idx == i
            if torch.any(select):
                v_zone = v[..., select].contiguous()
                batch_size = v_zone.shape[0]
                num_selected = v_zone.shape[-1]
                
                # Expand input
                v_expanded = v_zone.unsqueeze(0).expand(
                    KSTEPS, batch_size, 2, self.temporal_input, num_selected
                ).contiguous().reshape(
                    KSTEPS * batch_size, 2, self.temporal_input, num_selected
                )
                
                # Prepare noise for current zone
                noise_weight = self.implicit_cells[i].noise_w * self.noise_weight_tensor[i]
                noise_scaled = noise * noise_weight
                
                # Expand noise to match input dimensions
                noise_expanded = noise_scaled.expand(
                    KSTEPS, batch_size, 2, self.temporal_input, num_selected
                ).contiguous().reshape(
                    KSTEPS * batch_size, 2, self.temporal_input, num_selected
                )
                
                v_expanded = v_expanded + noise_expanded
                
                out = self.implicit_cells[i](v_expanded, weight_select=i)
                out = out.reshape(KSTEPS, batch_size, 2, self.temporal_output, num_selected)
                out = out.mean(dim=1)  # Average over batch dimension
                v_out[..., select] = out
                
        return v_out
