import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 这两个工具类来自你的项目：SinusoidalPosEmb 做时间/步数的正余弦位置编码；LayerNorm 是 NAFNet 风格的按通道-空间归一化
from model.sr3_modules.nafnet_util import SinusoidalPosEmb, LayerNorm


class SimpleGate(nn.Module):
    """
    SimpleGate：把通道均分为两半后逐点相乘，既当“非线性”也起到降通道的作用（2C -> C）。
    要求输入通道数为偶数（以便 chunk 成两份）。
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # (B, C, H, W) -> 两个 (B, C/2, H, W)
        return x1 * x2               # 逐点乘法门控，输出 (B, C/2, H, W)


class SSDA(nn.Module):
    """
    Simple Spatial Decomposition Attention (SSDA)
    用 1xK 与 Kx1 两个**深度卷积**分别建模水平方向/垂直方向的空间上下文，
    然后把两个响应逐点相乘得到 gate_map（空间注意力图）。
    最终在 NAFBlock 中：sigmoid(SCA + SSDA) 作为门控因子。
    """
    def __init__(self, c, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # 水平深度卷积（1×K），groups=c 说明每个通道各自卷积（depthwise）
        self.conv_h = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(1, kernel_size),
                                padding=(0, padding), groups=c, bias=True)

        # 垂直深度卷积（K×1）
        self.conv_v = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(kernel_size, 1),
                                padding=(padding, 0), groups=c, bias=True)

        # 如果需要再跟 1×1 融合，可打开下面的 fuse（这里保留为注释）
        # self.fuse_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, c, H, W)
        x_h = self.conv_h(x)     # (B, c, H, W)
        x_v = self.conv_v(x)     # (B, c, H, W)
        gate_map = x_h * x_v     # 空间逐点乘，得到方向互补的空间注意力图 (B, c, H, W)

        # 若希望直接把 gate_map 乘回特征并再融合，可启用下面两行
        # x = x * gate_map
        # x = self.fuse_conv(x)
        return gate_map          # 这里只返回注意力图，供上层 sigmoid(SCA + SSDA) 使用


class NAFBlock(nn.Module):
    """
    单个 NAF 块（带时间调制）
    - att 分支（空间建模）：1x1（扩通道） -> 3x3 深度卷积 -> SimpleGate -> SCA(+SSDA) 门控 -> 1x1（回通道） -> 残差
    - ffn 分支（通道混合）：1x1（扩） -> SimpleGate -> 1x1（回） -> 残差
    - 两处 FiLM 时间调制（在 att 和 ffn 进入处），分别用 (scale+1) 与 shift 做仿射：x*(scale+1)+shift
    """
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        # 块内用于把 time_emb 映射到 4c 的小 MLP：SimpleGate 先把 time_emb 对半乘法，再线性到 4c
        # 输出 4c 的含义：分成四份 (shift_att, scale_att, shift_ffn, scale_ffn)，每份 c 通道
        self.mlp = nn.Sequential(
            SimpleGate(),
            nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        # ---- att 分支（空间建模）用到的卷积 ----
        dw_channel = c * DW_Expand  # 扩通道到 2c（默认），以便后接 depthwise 3x3
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)  # 1×1, c -> 2c

        # depthwise 3x3：groups=dw_channel，表示每个通道各自卷积（极大降参/FLOPs）
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)  # 3×3 DW, 2c -> 2c

        # SimpleGate 之后通道从 2c 变回 c，因此这里的 in_channels 是 dw_channel // 2
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)  # 1×1, c -> c

        # ---- 简化通道注意力（SCA）----
        # 做 GAP 得到 (B, c, 1, 1)，再用 1×1 conv 生成通道注意力（未过激活，后面与 SSDA 相加再 sigmoid）
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, c, H, W) -> (B, c, 1, 1)
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)  # 保持通道 c 不变
        )

        self.sg = SimpleGate()        # 乘法门控：2c -> c
        self.ssda = SSDA(c=dw_channel // 2)  # 空间注意力（方向分解的 DW 卷积），输出 (B, c, H, W)

        # ---- ffn 分支（通道混合）用到的卷积 ----
        ffn_channel = FFN_Expand * c  # 默认 2c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)  # 1×1, c -> 2c
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)  # 1×1, c -> c

        # 前置 LayerNorm（NAF/Transformer 常见的 pre-norm 结构）
        self.norm1 = LayerNorm(c)  # 给 att 分支入口用
        self.norm2 = LayerNorm(c)  # 给 ffn 分支入口用

        # 可选 dropout（默认不开）
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 两个可学习残差缩放系数：初始为 0，有助于训练初期接近恒等映射更稳定
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)   # 对应 att 分支输出的残差
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)  # 对应 ffn 分支输出的残差

    def time_forward(self, time, mlp):
        """
        把 time_emb（形状 (B, time_emb_dim)）映射到 (B, 4c, 1, 1)，再切成4份
        分别对应：shift_att, scale_att, shift_ffn, scale_ffn（每份都是 (B, c, 1, 1)）
        """
        time_emb = mlp(time)                            # (B, 4c)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)                # 四份 (B, c, 1, 1)

    def forward(self, x):
        """
        x: tuple(inp, time)
           - inp: (B, c, H, W)    当前分辨率的特征图
           - time: (B, time_emb_dim) 由 Model.time_mlp 生成的时间嵌入
        return: (out, time)       与输入形式保持一致，便于上层串联调用
        """
        inp, time = x  # inp: (B, c, H, W), time: (B, time_emb_dim)

        # 生成四组 FiLM 参数（逐通道的 scale & shift）
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        # -------------------- att 分支（空间建模） --------------------
        x = inp
        x = self.norm1(x)                            # 预归一化（按通道-空间）
        x = x * (scale_att + 1) + shift_att          # FiLM(att)：逐通道仿射，(B,c,1,1) 自动广播到 H、W

        x = self.conv1(x)                            # 1×1 扩通道：c -> 2c
        x = self.conv2(x)                            # 3×3 深度卷积：2c -> 2c
        x = self.sg(x)                               # SimpleGate：2c -> c（乘法门控）

        # SCA（通道注意）输出 (B,c,1,1)；SSDA（空间注意）输出 (B,c,H,W)
        # 两者相加后 sigmoid，作为门控因子逐点相乘
        x = x * torch.sigmoid(self.sca(x) + self.ssda(x))

        x = self.conv3(x)                            # 1×1 回到 c 通道
        x = self.dropout1(x)

        # 残差 1（可学习缩放 beta 初值为 0）
        y = inp + x * self.beta                      # (B, c, H, W)

        # -------------------- ffn 分支（通道混合） --------------------
        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn          # FiLM(ffn)：逐通道仿射

        x = self.conv4(x)                            # 1×1：c -> 2c
        x = self.sg(x)                               # SimpleGate：2c -> c
        x = self.conv5(x)                            # 1×1：c -> c
        x = self.dropout2(x)

        # 残差 2（可学习缩放 gamma 初值为 0）
        x = y + x * self.gamma                       # (B, c, H, W)

        return x, time


class Model(nn.Module):
    """
    整体网络（U 形结构）：
    - 输入通道 6，输出通道 3
    - 编码器：每层若干 NAFBlock 后接一次 stride=2 下采样（通道×2，空间/2）
    - 中间块：最低分辨率处的若干 NAFBlock
    - 解码器：每层先 1×1 卷积扩到 2*chan 再 PixelShuffle(2) 上采样（空间×2，通道/2），与对称的 encoder 跳连相加，再过若干 NAFBlock
    - 时间条件：time_mlp 产出 (B, time_dim)，传给每个 NAFBlock 做两处 FiLM
    """
    def __init__(self):
        super().__init__()
        img_channel = 6
        width = 64                 # 基准通道数（第一层输出通道）；常用 64：容量-算力折中 & 对齐 PixelShuffle/SG 等形状
        middle_blk_num = 2         # 中间块堆叠数量
        enc_blk_nums = tuple([2, 2, 2, 2])  # 每个编码阶段 NAFBlock 的数量
        dec_blk_nums = tuple([2, 2, 2, 2])  # 每个解码阶段 NAFBlock 的数量
        upscale = 1
        self.upscale = upscale

        # ---- 时间嵌入（全局）----
        fourier_dim = width                # SinusoidalPosEmb 输出维度
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4               # 这里设为 4*width（=256），便于后续块内 MLP 映射到 4c

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,                  # t -> (B, fourier_dim) 的正余弦编码
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),                  # (2*time_dim) -> time_dim
            nn.Linear(time_dim, time_dim)  # 输出 (B, time_dim)
        )

        # 头尾卷积：映射输入到基准通道；尾部映射到 RGB 3 通道
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        # 编码器/解码器/上下采样/中间块容器
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # ---- 搭建编码器（4 层）----
        chan = width
        for num in enc_blk_nums:
            # 每层编码器：堆 num 个 NAFBlock(chan, time_dim)
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            # 下采样：stride=2 的 2x2 卷积，通道翻倍，空间减半
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2  # 更新通道数

        # ---- 中间块（最低分辨率处）----
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
        )

        # ---- 搭建解码器（4 层，对称）----
        for num in dec_blk_nums:
            # 上采样：先用 1×1 把通道变成 2*chan，再 PixelShuffle(2) -> 空间×2，通道/2
            # 注意：PixelShuffle(2) 要求输入通道可以整除 4，这里 2*chan 一定满足
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2  # 上采样后通道减半（与编码器对称）

            # 该阶段的若干 NAFBlock
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        # 需要 padding 的最小倍数（因为编码端做了 len(encoders) 次 /2）
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x, time):
        """
        x: (B, 6, H, W)
        time: (B,) 或标量；如果是标量，会被包装成 (1,) 并广播使用
        """
        # 兼容标量时间：统一成张量
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(x.device)

        # 全局时间嵌入（供所有 NAFBlock 使用）
        t = self.time_mlp(time)  # (B, time_dim)

        # 记录原始尺寸，因下面可能 padding 到 2^depth 的倍数
        B, C, H, W = x.shape
        x = self.check_image_size(x)  # 右/下 pad

        # 头部卷积，映射到 width 通道
        x = self.intro(x)

        # 编码：每层 encoder -> 保存 skip -> 下采样
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])  # 每个 NAFBlock 内部会使用 t 做 FiLM
            encs.append(x)          # 存跳连
            x = down(x)             # 下采样（通道×2，空间/2）

        # 中间块
        x, _ = self.middle_blks([x, t])

        # 解码：上采样 -> 加上对称的 skip -> decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)               # PixelShuffle 上采样（空间×2，通道/2）
            x = x + enc_skip        # 跳连融合（逐元素相加）
            x, _ = decoder([x, t])  # 再堆若干 NAFBlock

        # 尾部卷积：映射到 3 通道 RGB
        x = self.ending(x)

        # 去掉 padding，裁回原始 (H, W)
        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        """
        保证输入尺寸能被 2^depth 整除（因为编码器要重复 /2 下采样），
        通过在右/下两侧做零填充来实现；最后 forward 会裁回原尺寸。
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
