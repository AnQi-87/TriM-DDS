import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, SchNet
from torch_geometric.nn import global_max_pool as gmp
import torchvision
from losses import SupConLoss

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
        concat = torch.cat([feat1, feat2, feat3], dim=1)  
        weights = self.attn(concat)                       
        weights = F.softmax(weights, dim=1)               
        fused = weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2 + weights[:, 2:3] * feat3
        return fused, weights

class TriM_DDS(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=64, num_features_xt=954, output_dim=128, dropout=0.1,
                 use_cl=False, use_image_fusion=False, use_3d_fusion=False,
                 img_pretrained=True, temperature=0.07, base_temperature=0.07,
                 batch_size=128, device=torch.device('cpu')):
        super(TriM_DDS, self).__init__()
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
            print('Initializing Official PyG SchNet encoder...')
            self.encoder_3d = SchNet(
                hidden_channels=output_dim,  
                num_filters=output_dim,      
                num_interactions=3,          
                cutoff=10.0,                 
                readout='add'                
            )
            
            self.encoder_3d.lin1 = nn.Linear(output_dim, output_dim)
            self.encoder_3d.lin2 = nn.Linear(output_dim, output_dim)

        if self.use_img_fusion and self.use_3d_fusion:
            self.fusion = DynamicFusion3(dim=output_dim)
        elif self.use_img_fusion and not self.use_3d_fusion:
            self.fusion = AttentionFusion_auto(n_dim_input1=output_dim, n_dim_input2=output_dim)
        else:
            self.fusion = None

        self.final_mlp = nn.Sequential(
            nn.Linear(3 * output_dim, 512),
            nn.BatchNorm1d(512),
            self.activate,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activate,
            nn.Linear(256, n_output)
        )

        self.cl_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)

    def forward(self, data1, data2):
        x1, image1, edge_index1, batch1, cell = data1.x, data1.image, data1.edge_index, data1.batch, data1.cell
        x2, image2, edge_index2, batch2 = data2.x, data2.image, data2.edge_index, data2.batch

        # drug1
        x1_out = self.drug_conv1(x1, edge_index1)
        x1_out = self.activate(x1_out)
        x1_out = self.drug_conv2(x1_out, edge_index1)
        x1_out = self.activate(x1_out)
        x1_out = gmp(x1_out, batch1)
        x1_out = self.drug_fc_g1(x1_out)
        x1_out = self.activate(x1_out)
        x1_out = self.dropout(x1_out)
        x1_out = self.drug_fc_g2(x1_out)
        x1_out = self.dropout(x1_out)
        x1_graph = F.normalize(x1_out, dim=1)
        
        current_batch_size = x1_graph.shape[0]

        x1_image = None
        if self.use_img_fusion or self.use_cl:
            x1_image = self.image_model(image1)
            x1_image = F.normalize(x1_image, dim=1)

        x1_3d = None
        if self.use_3d_fusion or self.use_cl:
            if hasattr(data1, 'pos_3d'):
                pos1 = data1.pos_3d
                z1 = data1.z_3d
                num_nodes_1 = data1.num_nodes_3d
                
                batch_size_1 = num_nodes_1.size(0)
                batch1_3d = torch.repeat_interleave(
                    torch.arange(batch_size_1, device=pos1.device), 
                    num_nodes_1
                )
                
                x1_3d = self.encoder_3d(z1, pos1, batch1_3d)
                
                if x1_3d.dim() == 2 and x1_3d.shape[0] != batch_size_1:
                    x1_3d = gmp(x1_3d, batch1_3d)
                    
                x1_3d = F.normalize(x1_3d, dim=1)
            else:
                x1_3d = torch.zeros(current_batch_size, self.output_dim, device=x1_graph.device)

        if self.use_img_fusion and self.use_3d_fusion and self.fusion is not None:
            x1_fused, _ = self.fusion(x1_graph, x1_image, x1_3d)
        elif self.use_img_fusion and not self.use_3d_fusion and self.fusion is not None:
            x1_fused = self.fusion(x1_graph, x1_image)
        else:
            x1_fused = x1_graph

        # drug2
        x2_out = self.drug_conv1(x2, edge_index2)
        x2_out = self.activate(x2_out)
        x2_out = self.drug_conv2(x2_out, edge_index2)
        x2_out = self.activate(x2_out)
        x2_out = gmp(x2_out, batch2)
        x2_out = self.drug_fc_g1(x2_out)
        x2_out = self.activate(x2_out)
        x2_out = self.dropout(x2_out)
        x2_out = self.drug_fc_g2(x2_out)
        x2_out = self.dropout(x2_out)
        x2_graph = F.normalize(x2_out, dim=1)

        x2_image = None
        if self.use_img_fusion or self.use_cl:
            x2_image = self.image_model(image2)
            x2_image = F.normalize(x2_image, dim=1)

        x2_3d = None
        if self.use_3d_fusion or self.use_cl:
            if hasattr(data2, 'pos_3d'):
                pos2 = data2.pos_3d
                z2 = data2.z_3d
                num_nodes_2 = data2.num_nodes_3d
                
                batch_size_2 = num_nodes_2.size(0)
                batch2_3d = torch.repeat_interleave(
                    torch.arange(batch_size_2, device=pos2.device), 
                    num_nodes_2
                )
                
                x2_3d = self.encoder_3d(z2, pos2, batch2_3d)
                
                if x2_3d.dim() == 2 and x2_3d.shape[0] != batch_size_2:
                    x2_3d = gmp(x2_3d, batch2_3d)
                    
                x2_3d = F.normalize(x2_3d, dim=1)
            else:
                x2_3d = torch.zeros(current_batch_size, self.output_dim, device=x2_graph.device)

        if self.use_img_fusion and self.use_3d_fusion and self.fusion is not None:
            x2_fused, _ = self.fusion(x2_graph, x2_image, x2_3d)
        elif self.use_img_fusion and not self.use_3d_fusion and self.fusion is not None:
            x2_fused = self.fusion(x2_graph, x2_image)
        else:
            x2_fused = x2_graph

        # cell
        cell_vector = F.normalize(cell, dim=1)
        cell_vector = self.reduction(cell_vector)          

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