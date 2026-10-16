"""MIC — Model Microphone / 模型麦克风.

让任何中小型黑箱模型（金融 / 医学 / 地理 / 科学……）"开口说话"的通用审讯框架：

    模型驱动适配器  →  统一审讯 API  →  归因 / 反事实 / 干预式因果图
                        ↓                                  ↓
                   LLM 三角色智能体（翻译官 / 审查员 / 修复师）  忠实性验证层

设计原则（源自 MAIA / OpenAI auto-interp / Anthropic circuit tracing 的启示）：
LLM 的一切结论必须来自它调用的干预实验，而不是它的想象。
"""

__version__ = "0.1.0"
