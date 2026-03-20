import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_max_pool as gmp
import numpy as np
from losses import SupConLoss
import torchvision


class AttentionFusion_auto(torch.nn.Module):
    def __init__(self, n_dim_input1, n_dim_input2):
        super(AttentionFusion_auto, self).__init__()
        self.n_dim_input1, self.n_dim_input2 = n_dim_input1, n_dim_input2
        self.linear = nn.Linear(2 * n_dim_input1, n_dim_input2)

    def forward(self, input1, input2):
        mid_emb = torch.cat((input1, input2), 1)
        return F.relu(self.linear(mid_emb))


class DynamicFusion3(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(DynamicFusion3, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, feat1, feat2, feat3):
        concat = torch.cat([feat1, feat2, feat3], dim=1)  # (B, 3*dim)
        weights = self.attn(concat)                       # (B, 3)
        weights = F.softmax(weights, dim=1)               # (B, 3)
        
        fused = weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2 + weights[:, 2:3] * feat3
        return fused, weights


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, hidden_dim=256):
        super(PointNetEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        batch_size, N, D = x.shape
        x_flat = x.reshape(-1, D)                       # (batch*N, D)
        x_flat = self.mlp(x_flat)                        # (batch*N, out_dim)
        x_out = x_flat.reshape(batch_size, N, -1)        # (batch, N, out_dim)
        x_pool, _ = torch.max(x_out, dim=1)              # (batch, out_dim)
        return x_pool


class ImageDDS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=954, output_dim=128, dropout=0.1,
                 use_cl=False, use_image_fusion=False, use_3d_fusion=False,
                 img_pretrained=True, temperature=0.07, base_temperature=0.07,
                 batch_size=128, device=torch.device('cpu')):
        super(ImageDDS, self).__init__()
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_cl = use_cl
        self.use_img_fusion = use_image_fusion
        self.use_3d_fusion = use_3d_fusion
        self.output_dim = output_dim

        self.drug_conv1 = TransformerConv(78, num_features_xd * 2, heads=2)
        self.drug_conv2 = TransformerConv(num_features_xd * 4, num_features_xd * 8, heads=2)

        self.drug_fc_g1 = torch.nn.Linear(num_features_xd * 16, num_features_xd * 8)
        self.drug_fc_g2 = torch.nn.Linear(num_features_xd * 8, output_dim)

        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

        if self.use_img_fusion or self.use_cl or self.use_3d_fusion:
            print('Loading image model...')
            self.image_model = torchvision.models.resnet18(pretrained=img_pretrained)
            self.image_model.fc = nn.Linear(self.image_model.fc.in_features, output_dim)

        if self.use_3d_fusion or self.use_cl:
            print('Initializing 3D encoder...')
            self.pointnet_enc = PointNetEncoder(in_dim=3+78, out_dim=output_dim)

        print('Loading...')

        if self.use_img_fusion and self.use_3d_fusion:
            self.fusion = DynamicFusion3(dim=output_dim)
        elif self.use_img_fusion and not self.use_3d_fusion:
            self.fusion = AttentionFusion_auto(n_dim_input1=output_dim, n_dim_input2=output_dim)
        else:
            self.fusion = None

        self.final_mlp = nn.Sequential(
            nn.Linear(3 * output_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )

        self.cl_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)

    def forward(self, data1, data2):
        x1, image1, edge_index1, batch1, cell = data1.x, data1.image, data1.edge_index, data1.batch, data1.cell
        x2, image2, edge_index2, batch2 = data2.x, data2.image, data2.edge_index, data2.batch
        feat3d1 = data1.feat3d if hasattr(data1, 'feat3d') else None
        feat3d2 = data2.feat3d if hasattr(data2, 'feat3d') else None

        # drug1
        x1 = self.drug_conv1(x1, edge_index1)
        x1 = self.activate(x1)
        x1 = self.drug_conv2(x1, edge_index1)
        x1 = self.activate(x1)
        x1 = gmp(x1, batch1)
        x1 = self.drug_fc_g1(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)
        x1 = self.drug_fc_g2(x1)
        x1 = self.dropout(x1)
        x1_graph = F.normalize(x1, dim=1)

        current_batch_size = x1_graph.shape[0]


        x1_image = None
        if self.use_img_fusion or self.use_cl:
            x1_image = self.image_model(image1)
            x1_image = F.normalize(x1_image, dim=1)


        x1_3d = None
        if self.use_3d_fusion or self.use_cl:
            if feat3d1 is not None:
                if feat3d1.dim() == 2:
                    feat3d1 = feat3d1.unsqueeze(0)
                if feat3d1.shape[0] != current_batch_size:
                    feat3d1 = feat3d1.expand(current_batch_size, -1, -1).contiguous()
                x1_3d = self.pointnet_enc(feat3d1)
                x1_3d = F.normalize(x1_3d, dim=1)
            else:
                x1_3d = torch.zeros(current_batch_size, self.output_dim, device=x1_graph.device)

        # Fusion
        if self.use_img_fusion and self.use_3d_fusion and self.fusion is not None:
            x1_fused, _ = self.fusion(x1_graph, x1_image, x1_3d) 
        elif self.use_img_fusion and not self.use_3d_fusion and self.fusion is not None:
            x1_fused = self.fusion(x1_graph, x1_image)            
        else:
            x1_fused = x1_graph

        # drug2
        x2 = self.drug_conv1(x2, edge_index2)
        x2 = self.activate(x2)
        x2 = self.drug_conv2(x2, edge_index2)
        x2 = self.activate(x2)
        x2 = gmp(x2, batch2)
        x2 = self.drug_fc_g1(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        x2 = self.drug_fc_g2(x2)
        x2 = self.dropout(x2)
        x2_graph = F.normalize(x2, dim=1)

        x2_image = None
        if self.use_img_fusion or self.use_cl:
            x2_image = self.image_model(image2)
            x2_image = F.normalize(x2_image, dim=1)

        x2_3d = None
        if self.use_3d_fusion or self.use_cl:
            if feat3d2 is not None:
                if feat3d2.dim() == 2:
                    feat3d2 = feat3d2.unsqueeze(0)
                if feat3d2.shape[0] != current_batch_size:
                    feat3d2 = feat3d2.expand(current_batch_size, -1, -1).contiguous()
                x2_3d = self.pointnet_enc(feat3d2)
                x2_3d = F.normalize(x2_3d, dim=1)
            else:
                x2_3d = torch.zeros(current_batch_size, self.output_dim, device=x2_graph.device)

        if self.use_img_fusion and self.use_3d_fusion and self.fusion is not None:
            x2_fused, _ = self.fusion(x2_graph, x2_image, x2_3d)
        elif self.use_img_fusion and not self.use_3d_fusion and self.fusion is not None:
            x2_fused = self.fusion(x2_graph, x2_image)
        else:
            x2_fused = x2_graph

        # cell line
        cell_vector = F.normalize(cell, dim=1)
        cell_vector = self.reduction(cell_vector)          # (batch, output_dim)

        # concat
        xc = torch.cat([x1_fused, x2_fused, cell_vector], dim=1)
        xc = F.normalize(xc, dim=1)
        final = self.final_mlp(xc)
        outputs = torch.sigmoid(final.squeeze(1))

        data_dict = {
            "img_idx_1": data1.img_idx,
            "x1_graph": x1_graph,
            "x1_image": x1_image,
            "x1_3d": x1_3d,
            "img_idx_2": data2.img_idx,
            "x2_graph": x2_graph,
            "x2_image": x2_image,
            "x2_3d": x2_3d
        }
        return outputs, data_dict

    def cal_cl_loss(self, data_dict):
        views1 = [data_dict["x1_graph"]]
        views2 = [data_dict["x2_graph"]]

        if self.use_img_fusion or self.use_cl:
            if data_dict["x1_image"] is not None:
                views1.append(data_dict["x1_image"])
            if data_dict["x2_image"] is not None:
                views2.append(data_dict["x2_image"])

        if self.use_3d_fusion or self.use_cl:
            if data_dict["x1_3d"] is not None:
                views1.append(data_dict["x1_3d"])
            if data_dict["x2_3d"] is not None:
                views2.append(data_dict["x2_3d"])
        
        if len(views1) < 2:
            loss1 = torch.tensor(0.0, device=data_dict["x1_graph"].device)
        else:
            loss1 = self.cl_loss(*views1, labels=data_dict["img_idx_1"])
            
        if len(views2) < 2:
            loss2 = torch.tensor(0.0, device=data_dict["x2_graph"].device)
        else:
            loss2 = self.cl_loss(*views2, labels=data_dict["img_idx_2"])
            
        return (loss1 + loss2) / 2