from strhub.models.baseline.system import CSLRBaselineSystem
from strhub.models.baseline_rm_sadecode.system import CSLRBaselineSystem as CSLRBaselineSystemIndependent
from strhub.models.baseline_linear.system import CSLRBaselineSystem as BaselineAttentionLinear
from strhub.models.baseline_linear.system import CSLRPositionSpecificSystem 
from strhub.models.baseGCN.system import GCNBaselineSystem
from strhub.models.baseline_autoregressive_permutate.system import CSLRAutoregressiveSystemPermutate
from strhub.models.baseline_autoregressive.system import CSLRAutoregressiveSystem
from strhub.models.baseline_gnn_autoregressive_permutate.system import GNN_AR_System
from strhub.models.region_focus_gnn_ar.system import RegionGNNCSLRSystem
from strhub.models.region_focus_gnn_ar_multiloss.system import MultiRegionCSLRSystem as RegionGNNCSLRSystem_MultiSupervision
from strhub.models.us_region_focus_gnn_multiloss.system import MultiRegionCSLRSystem as RegionGNNCSLRSystem_US
from strhub.models.region_focus_gnn_CTC.system import RegionGNNCSLRSystemCTC
from strhub.models.region_focus_gnn_ar_multiloss_rgb.system import MultiRegionCSLRSystemRGB
from strhub.models.region_focus_gnn_ar_multiloss_skeleton.system import MultiRegionCSLRSystemSkeleton
from strhub.models.region_focus_gnn_nar_us.system import NARRegionGNNCSLRSystem
from strhub.models.CTC_NAR_RegionGNN.system import CTC_NAR_RegionGNNSystem
from strhub.models.region_focus_gnn_CTC_2step.system import RegionGNNCSLRSystemCTC2Phase
from strhub.models.us_region_2stages.system import RegionGNNSystem2Stage
from strhub.models.region_focus_gnn_ar_multiloss_2loss.system import MultiRegionCSLRSystemVersion2
from strhub.models.region_focus_gnn_ar_version3.system import MultiRegionCSLRSystemVersion3

# Model system
MODEL_SYSTEMS = {
    "CSLRTransformerBaseline": CSLRBaselineSystem,
    "CSLRTransformerBaselineIndependent": CSLRBaselineSystemIndependent,
    "GCNTransformerBaseline": GCNBaselineSystem,
    "CSLRTransformerAutoregressivePermutate": CSLRAutoregressiveSystemPermutate,
    "CSLRTransformerAutoregressive": CSLRAutoregressiveSystem,
    "GNN_AR": GNN_AR_System,
    "BaselineAttentionLinear": BaselineAttentionLinear,
    "CSLRPositionSpecificSystem": CSLRPositionSpecificSystem,
    "RegionGNNCSLRSystem": RegionGNNCSLRSystem,
    "RegionGNNCSLRSystem_MultiSupervision": RegionGNNCSLRSystem_MultiSupervision,
    "RegionGNNCSLRSystem_US": RegionGNNCSLRSystem_US,
    "RegionGNNCSLRSystemCTC": RegionGNNCSLRSystemCTC,
    "RegionGNNCSLRSystemRGB": MultiRegionCSLRSystemRGB,
    "RegionGNNCSLRSystemSkeleton": MultiRegionCSLRSystemSkeleton,
    "NARRegionGNNCSLRSystem": NARRegionGNNCSLRSystem,
    "CTC_NAR_RegionGNNSystem": CTC_NAR_RegionGNNSystem,
    "RegionGNNCSLRSystemCTC2Phase": RegionGNNCSLRSystemCTC2Phase,
    "MultiRegionCSLRSystemVersion2": MultiRegionCSLRSystemVersion2,
    "MultiRegionCSLRSystemVersion3": MultiRegionCSLRSystemVersion3,
    "RegionGNNSystem2Stage": RegionGNNSystem2Stage
}