"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .LSTM1 import LSTM1
from .LSTM2 import LSTM2
from .LSTM3 import LSTM3
from .LSTM4 import LSTM4


class DCEM(nn.Module):
    def __init__(self, args):
        super(DCEM, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        self.dataset_name = args.dataset_name
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.need_data_aligned = args.need_data_aligned
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
                self.un_len_l, self.un_len_v, self.un_len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
                self.un_len_l, self.un_len_v, self.un_len_a = 50, 500, 500
        if args.dataset_name == 'sims':
            self.len_l, self.len_v, self.len_a = 50, 50, 50
            self.un_len_l, self.un_len_v, self.un_len_a = 39, 55, 400
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims # [768, 74, 35]
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_l = args.attn_dropout_l
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.out_dropout = args.out_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.logit_dims = 3
        self.fusion_dim = args.fusion_dim
        combined_dim_low = self.d_a
        combined_dim_high = self.d_a
        combined_dim = (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1
        self.combined_dim_low = combined_dim_low

        if args.dataset_name == 'mosi' and args.need_data_aligned is True:
            self.L_BiLSTM = LSTM1((self.len_l - args.conv1d_kernel_size + 1), 1)
            self.V_BiLSTM = LSTM1((self.len_v - args.conv1d_kernel_size_v + 1), 1)
            self.A_BiLSTM = LSTM1((self.len_a - args.conv1d_kernel_size_a + 1), 1)
        elif args.dataset_name == 'mosi' and args.need_data_aligned is False:
            self.L_BiLSTM = LSTM2(self.un_len_l, self.len_l, 1, args.conv1d_kernel_size - 1, layer_norm=False)
            self.V_BiLSTM = LSTM2(self.un_len_v, self.len_v, 1, args.conv1d_kernel_size_v - 1, layer_norm=False)
            self.A_BiLSTM = LSTM2(self.un_len_a, self.len_a, 1, args.conv1d_kernel_size_a - 1, layer_norm=False)
        elif args.dataset_name == 'mosei' and args.need_data_aligned is True:
            self.L_BiLSTM = LSTM1((self.len_l - args.conv1d_kernel_size + 1), 1)
            self.V_BiLSTM = LSTM3((self.len_v - args.conv1d_kernel_size_v + 1), 1, args.conv1d_kernel_size_v)
            self.A_BiLSTM = LSTM3((self.len_a - args.conv1d_kernel_size_a + 1), 1, args.conv1d_kernel_size_a)
        elif args.dataset_name == 'mosei' and args.need_data_aligned is False:
            self.L_BiLSTM = LSTM4(self.un_len_l, self.len_l, 1, args.conv1d_kernel_size - 1)
            self.V_BiLSTM = LSTM4(self.un_len_v, self.len_v, 1, args.conv1d_kernel_size_v - 1)
            self.A_BiLSTM = LSTM4(self.un_len_a, self.len_a, 1, args.conv1d_kernel_size_a - 1)

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * self.len_l,
                                       combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * self.len_v,
                                       combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * self.len_a,
                                       combined_dim_low)

        self.align_c_l = nn.Linear(combined_dim_low * self.len_l, combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * self.len_v, combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * self.len_a, combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='ll')
        self.self_attentions_c_v = self.get_network(self_type='vv')
        self.self_attentions_c_a = self.get_network(self_type='aa')

        self.self_attentions_s_l = self.get_network(self_type='l')
        self.self_attentions_s_v = self.get_network(self_type='v')
        self.self_attentions_s_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.logit_dims, self.fusion_dim)
        self.proj2_c = nn.Linear(self.fusion_dim, self.logit_dims)
        self.out_layer_c = nn.Linear(self.logit_dims, output_dim)

        self.proj1_s = nn.Linear(self.logit_dims, self.fusion_dim)
        self.proj2_s = nn.Linear(self.fusion_dim, self.logit_dims)
        self.out_layer_s = nn.Linear(self.logit_dims, output_dim)

        self.LayerNorm = nn.LayerNorm(dst_feature_dims)
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        self.proj1_l_low = nn.Linear(combined_dim_low * self.len_l, combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * self.len_l)
        self.out_layer_l_low = nn.Linear(combined_dim_low * self.len_l, output_dim)

        self.proj1_v_low = nn.Linear(combined_dim_low * self.len_v, combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * self.len_v)
        self.out_layer_v_low = nn.Linear(combined_dim_low * self.len_v, output_dim)

        self.proj1_a_low = nn.Linear(combined_dim_low * self.len_a, combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * self.len_a)
        self.out_layer_a_low = nn.Linear(combined_dim_low * self.len_a, output_dim)

        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)

        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)

        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        self.weight_l = nn.Linear(self.d_l, self.d_l)
        self.weight_v = nn.Linear(self.d_v, self.d_v)
        self.weight_a = nn.Linear(self.d_a, self.d_a)

        self.weight_cross = nn.Linear(12 * self.d_l, 12 * self.d_l)
        self.weight_c = nn.Linear(self.logit_dims, self.logit_dims)
        self.weight_s = nn.Linear(self.logit_dims, self.logit_dims)

        self.proj1 = nn.Linear(combined_dim * 2, combined_dim * 2)
        self.proj2 = nn.Linear(combined_dim * 2, combined_dim * 2)
        self.out_layer = nn.Linear(combined_dim * 2, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout_l
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        elif self_type == 'll':
            embed_dim, attn_dropout = 4 * self.d_l, self.attn_dropout
        elif self_type == 'aa':
            embed_dim, attn_dropout = 4 * self.d_a, self.attn_dropout
        elif self_type == 'vv':
            embed_dim, attn_dropout = 4 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        x_v = self.V_BiLSTM(x_v)
        x_a = self.A_BiLSTM(x_a)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)
        c_list = [c_l, c_v, c_a]

        c_l_sim = self.align_c_l(c_l.contiguous().view(c_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(c_v.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(c_a.size(0), -1))

        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        s_l_r = self.encoder_s_l(recon_l)
        s_v_r = self.encoder_s_v(recon_v)
        s_a_r = self.encoder_s_a(recon_a)


        att_cross_l = torch.cat([s_l, c_l, c_l, c_l], dim=1)
        att_cross_v = torch.cat([s_v, c_v, c_v, c_v], dim=1)
        att_cross_a = torch.cat([s_a, c_a, c_a, c_a], dim=1)

        att_cross_l = att_cross_l.permute(2, 0, 1)
        att_cross_v = att_cross_v.permute(2, 0, 1)
        att_cross_a = att_cross_a.permute(2, 0, 1)

        cross_l = torch.cat([c_l, s_l, s_a, s_v], dim=1)
        cross_v = torch.cat([c_v, s_v, s_a, s_l], dim=1)
        cross_a = torch.cat([c_a, s_a, s_v, s_l], dim=1)

        cross_l = cross_l.permute(2, 0, 1)
        cross_v = cross_v.permute(2, 0, 1)
        cross_a = cross_a.permute(2, 0, 1)

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        proj_s_l = self.proj_cosine_l(s_l.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_v = self.proj_cosine_v(s_v.transpose(0, 1).contiguous().view(x_l.size(0), -1))
        proj_s_a = self.proj_cosine_a(s_a.transpose(0, 1).contiguous().view(x_l.size(0), -1))

        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        c_fusion = torch.cat([logits_l_low, logits_v_low, logits_a_low], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.out_dropout,
                      training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)

        h_l = self.self_attentions_s_l(s_l)

        if type(h_l) == tuple:
            last_h_l = h_l[0]
        last_h_l = h_l[-1]

        h_v = self.self_attentions_s_v(s_v)
        if type(h_v) == tuple:
            last_h_v = h_v[0]
        last_h_v = h_v[-1]

        h_a = self.self_attentions_s_a(s_a)
        if type(h_a) == tuple:
            last_h_a = h_a[0]
        last_h_a = h_a[-1]

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        s_fusion = torch.cat([logits_l_high, logits_v_high, logits_a_high], dim=1)
        s_fusion = self.weight_s(s_fusion)
        s_fusion = torch.sigmoid(s_fusion)
        s_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(s_fusion), inplace=True), p=self.out_dropout,
                      training=self.training))
        s_proj += s_fusion

        logits_s = self.out_layer_s(s_proj)

        c_l_att = self.self_attentions_c_l(att_cross_l, att_cross_l, cross_l)
        c_v_att = self.self_attentions_c_v(att_cross_v, att_cross_v, cross_v)
        c_a_att = self.self_attentions_c_a(att_cross_a, att_cross_a, cross_a)

        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]
        # print(c_l_att.shape)

        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
        fusion = torch.sigmoid(self.weight_cross(fusion))

        last_hs = fusion
        last_hs = last_hs.view((last_hs.shape[0], -1))
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'proj_s_l': proj_s_l,
            'proj_s_v': proj_s_v,
            'proj_s_a': proj_s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'last_h_l': h_l[-1],
            'last_h_v': h_v[-1],
            'last_h_a': h_a[-1],
            'logits_c': logits_c,
            'logits_s': logits_s,
            'output_logit': output,
        }
        return res