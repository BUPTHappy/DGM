from unified_video_action.model.ucgm.transports.edm import EDM
from unified_video_action.model.ucgm.transports.linear import Linear
from unified_video_action.model.ucgm.transports.random import Random
from unified_video_action.model.ucgm.transports.relinear import ReLinear
from unified_video_action.model.ucgm.transports.trigflow import TrigFlow
from unified_video_action.model.ucgm.transports.triglinear import TrigLinear
from unified_video_action.model.ucgm.transports.cosine import Cosine
from unified_video_action.model.ucgm.transports.ddpm import DDPM


TRANSPORTS = {
    "EDM": EDM,
    "Linear": Linear,
    "Random": Random,
    "ReLinear": ReLinear,
    "TrigFlow": TrigFlow,
    "TrigLinear": TrigLinear,
    "Cosine": Cosine,
    "DDPM": DDPM
}
